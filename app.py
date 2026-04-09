"""
音声議事録ツール — Render中継サーバー

kintone JSから音声/動画データ(Base64)を受け取り、
ffmpegで音声抽出＆圧縮 → Whisper APIで文字起こし → Claude APIで議事録整形 → 結果を返す
"""

import os
import base64
import tempfile
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import imageio_ffmpeg

app = Flask(__name__)
CORS(app)

# 大きいファイル対応：最大500MB（Base64込み）
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# ffmpegバイナリパス
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

MINUTES_PROMPT = """以下の文字起こしテキストから議事録を作成してください。

## 出力フォーマット
- 日時
- 参加者（判別できる場合）
- 議題・目的
- 内容の要約
- 決定事項
- 次のアクション（TODO）
- 備考

判別できない項目は「記載なし」としてください。

## 文字起こしテキスト
"""


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({'success': False, 'error': '音声データが送信されていません'}), 400

        audio_base64 = data['audio']
        file_name = data.get('fileName', 'audio.m4a')
        mime_type = data.get('mimeType', 'audio/m4a')

        # Base64デコード
        audio_bytes = base64.b64decode(audio_base64)
        original_mb = len(audio_bytes) / (1024 * 1024)
        app.logger.info(f'受信ファイル: {file_name} ({original_mb:.1f}MB)')

        # Step 0: ffmpegで音声抽出＆圧縮（動画→音声、音声→低ビットレート化）
        compressed_path = compress_audio(audio_bytes, file_name)

        try:
            compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
            app.logger.info(f'圧縮後: {compressed_size:.1f}MB')

            # 圧縮後も25MB超ならエラー
            if compressed_size > 25:
                return jsonify({
                    'success': False,
                    'error': f'圧縮後もファイルサイズが25MBを超えます（{compressed_size:.1f}MB）。録音時間が長すぎる可能性があります。'
                }), 400

            # Step 1: Whisper APIで文字起こし
            transcript = transcribe_with_whisper(compressed_path)

            # Step 2: Claude APIで議事録整形
            minutes = generate_minutes_with_claude(transcript)

            return jsonify({
                'success': True,
                'transcript': transcript,
                'minutes': minutes
            })
        finally:
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)

    except Exception as e:
        app.logger.exception('処理エラー')
        return jsonify({'success': False, 'error': f'エラーが発生しました: {str(e)}'}), 500


def compress_audio(audio_bytes, file_name):
    """ffmpegで音声を抽出＆圧縮（動画ファイルにも対応）

    - 動画ファイル：音声トラックを抽出
    - 全ファイル：モノラル、16kHzサンプリング、24kbpsのmp3に変換
    - 1時間の会議録音でも約10MB程度になる
    """
    ext = file_name.rsplit('.', 1)[-1].lower() if '.' in file_name else 'bin'

    # 入力ファイルを一時保存
    with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name

    # 出力ファイルパス
    output_path = tempfile.mktemp(suffix='.mp3')

    try:
        cmd = [
            FFMPEG_PATH,
            '-i', input_path,
            '-vn',                  # 映像トラックを無視
            '-ac', '1',             # モノラル
            '-ar', '16000',         # 16kHz（音声認識に十分）
            '-b:a', '24k',          # 24kbps（低ビットレート）
            '-f', 'mp3',
            '-y',                   # 上書き許可
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)

        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')[:500]
            raise Exception(f'音声変換に失敗しました: {error_msg}')

        return output_path
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


def transcribe_with_whisper(audio_path):
    """Whisper APIで文字起こし"""
    if not OPENAI_API_KEY:
        raise Exception('OPENAI_API_KEYが設定されていません')

    with open(audio_path, 'rb') as audio_file:
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
            files={'file': ('audio.mp3', audio_file, 'audio/mpeg')},
            data={'model': 'whisper-1', 'language': 'ja'},
            timeout=600
        )

    if response.status_code != 200:
        raise Exception(f'Whisper API エラー (HTTP {response.status_code}): {response.text[:300]}')

    return response.json()['text']


def generate_minutes_with_claude(transcript):
    """Claude APIで議事録整形"""
    if not ANTHROPIC_API_KEY:
        raise Exception('ANTHROPIC_API_KEYが設定されていません')

    response = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 4096,
            'messages': [
                {'role': 'user', 'content': MINUTES_PROMPT + transcript}
            ]
        },
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f'Claude API エラー (HTTP {response.status_code}): {response.text[:300]}')

    return response.json()['content'][0]['text']


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

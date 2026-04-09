"""
音声議事録ツール — Render中継サーバー

kintone JSからGoogle DriveのURLを受け取り、
ファイルをダウンロード → ffmpeg圧縮 → Whisper文字起こし → Claude議事録整形 → 結果を返す
"""

import os
import re
import tempfile
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import imageio_ffmpeg
import gdown

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

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
    """Google DriveのURLを受け取り、議事録を生成"""
    download_path = None
    compressed_path = None

    try:
        data = request.get_json()
        if not data or 'driveUrl' not in data:
            return jsonify({'success': False, 'error': 'Google DriveのURLが指定されていません'}), 400

        drive_url = data['driveUrl'].strip()

        # Drive ファイルIDを抽出
        file_id = extract_drive_file_id(drive_url)
        if not file_id:
            return jsonify({
                'success': False,
                'error': 'Google DriveのURLが認識できません。共有リンクをそのまま貼り付けてください。'
            }), 400

        app.logger.info(f'Drive file ID: {file_id}')

        # Step 1: Driveからダウンロード
        download_path = tempfile.mktemp(suffix='.bin')
        try:
            gdown.download(id=file_id, output=download_path, quiet=True, fuzzy=True)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Google Driveからのダウンロードに失敗しました。共有設定を「リンクを知っている全員」に変更してください。'
            }), 400

        if not os.path.exists(download_path) or os.path.getsize(download_path) == 0:
            return jsonify({
                'success': False,
                'error': 'ファイルが取得できませんでした。共有設定を確認してください。'
            }), 400

        original_mb = os.path.getsize(download_path) / (1024 * 1024)
        app.logger.info(f'ダウンロード完了: {original_mb:.1f}MB')

        # Step 2: ffmpegで音声抽出＆圧縮
        compressed_path = compress_audio(download_path)
        compressed_mb = os.path.getsize(compressed_path) / (1024 * 1024)
        app.logger.info(f'圧縮後: {compressed_mb:.1f}MB')

        if compressed_mb > 25:
            return jsonify({
                'success': False,
                'error': f'圧縮後もファイルサイズが25MBを超えます（{compressed_mb:.1f}MB）。録音時間が長すぎる可能性があります。'
            }), 400

        # Step 3: Whisper APIで文字起こし
        transcript = transcribe_with_whisper(compressed_path)

        # Step 4: Claude APIで議事録整形
        minutes = generate_minutes_with_claude(transcript)

        return jsonify({
            'success': True,
            'transcript': transcript,
            'minutes': minutes
        })

    except Exception as e:
        app.logger.exception('処理エラー')
        return jsonify({'success': False, 'error': f'エラーが発生しました: {str(e)}'}), 500

    finally:
        for p in [download_path, compressed_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


def extract_drive_file_id(url):
    """Google Drive URLからファイルIDを抽出

    対応形式:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID
    - FILE_ID（IDのみ）
    """
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)',
        r'^([a-zA-Z0-9_-]{25,})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def compress_audio(input_path):
    """ffmpegで音声を抽出＆圧縮（動画ファイルにも対応）

    モノラル、16kHzサンプリング、24kbpsのmp3に変換
    """
    output_path = tempfile.mktemp(suffix='.mp3')

    cmd = [
        FFMPEG_PATH,
        '-i', input_path,
        '-vn',                  # 映像トラックを無視
        '-ac', '1',             # モノラル
        '-ar', '16000',         # 16kHz
        '-b:a', '24k',          # 24kbps
        '-f', 'mp3',
        '-y',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)

    if result.returncode != 0:
        error_msg = result.stderr.decode('utf-8', errors='ignore')[:500]
        raise Exception(f'音声変換に失敗しました: {error_msg}')

    return output_path


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

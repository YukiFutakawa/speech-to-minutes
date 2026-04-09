"""
音声議事録ツール — Render中継サーバー

kintone JSから音声データ(Base64)を受け取り、
Whisper APIで文字起こし → Claude APIで議事録整形 → 結果を返す
"""

import os
import base64
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

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

        # サイズチェック（25MB）
        size_mb = len(audio_bytes) / (1024 * 1024)
        if size_mb > 25:
            return jsonify({
                'success': False,
                'error': f'ファイルサイズが25MBを超えています（{size_mb:.1f}MB）。25MB以下の音声ファイルを使用してください。'
            }), 400

        # Step 1: Whisper APIで文字起こし
        transcript = transcribe_with_whisper(audio_bytes, file_name, mime_type)

        # Step 2: Claude APIで議事録整形
        minutes = generate_minutes_with_claude(transcript)

        return jsonify({
            'success': True,
            'transcript': transcript,
            'minutes': minutes
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'エラーが発生しました: {str(e)}'}), 500


def transcribe_with_whisper(audio_bytes, file_name, mime_type):
    """Whisper APIで文字起こし"""
    if not OPENAI_API_KEY:
        raise Exception('OPENAI_API_KEYが設定されていません')

    # 一時ファイルに書き出してWhisper APIに送信
    ext = file_name.rsplit('.', 1)[-1] if '.' in file_name else 'm4a'
    with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'rb') as audio_file:
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
                files={'file': (file_name, audio_file, mime_type)},
                data={'model': 'whisper-1', 'language': 'ja'},
                timeout=300  # 5分タイムアウト
            )

        if response.status_code != 200:
            raise Exception(f'Whisper API エラー (HTTP {response.status_code}): {response.text}')

        return response.json()['text']
    finally:
        os.unlink(tmp_path)


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
        timeout=120  # 2分タイムアウト
    )

    if response.status_code != 200:
        raise Exception(f'Claude API エラー (HTTP {response.status_code}): {response.text}')

    return response.json()['content'][0]['text']


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

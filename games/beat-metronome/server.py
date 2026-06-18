import io
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없어요'}), 400

    file = request.files['file']
    audio_bytes = file.read()

    try:
        # librosa로 로드 (mono, 22050Hz로 자동 리샘플링)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)

        # 템포 힌트: 60~180 BPM 범위에서 탐색
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            start_bpm=120,
            tightness=100,   # 높을수록 일정한 템포 가정, 낮을수록 변화에 유연
            trim=False
        )

        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # BPM이 실제보다 절반으로 잡혔을 때 보정 (2/4 → 4/4 문제)
        if len(beat_times) > 1:
            avg_interval = np.mean(np.diff(beat_times[:20]))
            detected_bpm = 60 / avg_interval
            # BPM이 너무 낮으면 (60 미만) 비트 사이에 중간점 삽입
            if detected_bpm < 70:
                interp = []
                for i in range(len(beat_times) - 1):
                    interp.append(beat_times[i])
                    interp.append((beat_times[i] + beat_times[i+1]) / 2)
                interp.append(beat_times[-1])
                beat_times = interp

        return jsonify({
            'beats': beat_times,
            'bpm': round(60 / np.mean(np.diff(beat_times[:20])), 1)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('서버 시작: http://localhost:5000')
    app.run(port=5000, debug=False)

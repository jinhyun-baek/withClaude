import io
import os
import tempfile

# ── madmom/BeatNet이 의존하는 구식 API들을 최신 Python/numpy에서도 동작하도록 패치 ──
import collections
import collections.abc
for _name in ['MutableSequence', 'MutableMapping', 'Mapping', 'Sequence', 'Iterable']:
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

import numpy as np
for _name, _alias in [('float', float), ('int', int), ('bool', bool),
                       ('object', object), ('str', str), ('complex', complex)]:
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

import warnings
warnings.filterwarnings('ignore')

import librosa
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from BeatNet.BeatNet import BeatNet

app = Flask(__name__)
CORS(app)

print('BeatNet 모델 로딩 중...')
estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
print('BeatNet 준비 완료')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없어요'}), 400

    file = request.files['file']
    audio_bytes = file.read()
    tmp_path = None

    try:
        # BeatNet은 파일 경로가 필요해서 22050Hz 모노 WAV로 변환 후 임시 저장
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, y, sr)

        # BeatNet 추론: [[시간, 마디내박위치], ...] — 박위치 1이 다운비트(강박)
        output = estimator.process(tmp_path)
        beat_times = output[:, 0].tolist()
        beat_positions = output[:, 1].tolist()

        # 다운비트(1)가 시작되는 인덱스로 강박 위상 계산
        phase = 0
        for i, pos in enumerate(beat_positions):
            if round(pos) == 1:
                phase = i
                break

        # 감지된 박자표 (마디 내 최대 박 번호)
        time_sig = int(max(round(p) for p in beat_positions)) if beat_positions else 4

        # 평균 BPM 계산
        bpm = 120.0
        if len(beat_times) > 1:
            n = min(len(beat_times) - 1, 20)
            avg_interval = np.mean(np.diff(beat_times[:n + 1]))
            bpm = 60 / avg_interval

        return jsonify({
            'beats': beat_times,
            'bpm': round(bpm, 1),
            'phase': phase,
            'timeSig': time_sig
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == '__main__':
    print('서버 시작: http://localhost:5000')
    app.run(port=5000, debug=False)

# 프로젝트: SenseNova-U1.5 화풍 LoRA 학습

특정 게임 화풍(SD 비율 아바타 캐릭터) 데이터셋으로 SenseNova-U1.5 text-to-image **LoRA**를 학습한다.
추론 속도가 중요하므로 최종 사용은 **8-step 증류 LoRA(V2) + 화풍 LoRA, 8 steps / cfg 1.0**.
상세 배경, 근거, 결과: `SENSENOVA_8STEP_LORA_NOTES.md` (맨 위 "결과 요약"이 최신) · 학습 실행 방법: `sn15_train/README.md`

## 현재 상태 (2026-10-14)
- 학습 완료. **B안 채택**: 베이스에서 학습한 화풍 LoRA를 추론 시 8-step LoRA와 함께 붙인다.
  - 체크포인트: `DiffSynth-Studio/models/train/sn15_style_B/step-*.safetensors` (500 단위, 마지막 10080)
- A안(8-step LoRA를 fuse한 뒤 학습)은 이미지가 흐릿·반투명해져 실패했다. 8-step 모델 위에서 일반 flow-matching으로 학습하지 않는다.
- 추론: 루트의 `inference.py`. `PROMPT`, `STEP8`(True: 8-step/cfg 1, False: 50-step/cfg 4), `STYLE_STEP` 등을 파일 맨 위에서 바꾼다. 결과는 `outputs/`(gitignore).
- 남은 일:
  - 체크포인트 최종 선택(seed 여러 개로 비교)
  - 50-step 모드 품질 확인
  - A 체크포인트와 시험 학습 출력(약 5.2GB) 정리 여부 결정

## 결정 사항
- 트리거 단어 없음. 원본 캡션(의상 묘사)을 그대로 쓴다. 이 LoRA는 이 화풍 생성에만 쓴다.
- 평가용 데이터를 떼지 않는다. 560장 전부 학습하고, 평가는 새로 쓴 프롬프트로 한다.
  - 목표는 화풍과 프롬프트 충실도이지 정답 재현이 아니다.
- A/B 비교 → B 채택:
  - A = 학습 시 8-step LoRA를 fuse(`--preset_lora_path`) → 실패
  - B = 베이스에서만 학습하고 추론 시 stack → 채택
  - 둘 다 8-step으로 평가했다
- 학습과 GPU 작업은 사용자가 직접 실행한다. Claude는 스크립트와 명령 준비까지만 한다(요청이 있을 때만 실행).
- 서버는 ssh로 원격 접속한다. 긴 작업은 `sn15_train/tmux_start.sh <job>`으로 tmux 세션 `sn15`에서 돌린다.

## 구조
- `DiffSynth-Studio/`: upstream 코드(v2.1.9). 수정하지 않는다. 가중치는 `DiffSynth-Studio/models/`(gitignore).
- `dataset/<NFD 한글 폴더>/`: 원본 png/txt 560쌍.
  - 폴더명이 NFD라 타이핑하면 경로를 못 찾는다. `sn15_train/data` 심볼릭 링크나 glob을 쓴다.
- `sn15_train/`: 학습 키트.
  - 학습 설정은 `train.sh`에 있다: 10,080 step(18 epoch), 500 step마다 저장.
  - `train_sn15.py`는 공식 train.py에 체크포인트 시점 8-step 샘플 생성만 추가한 것이다.
  - 그 외: prepare_metadata.py, smoke.sh, run_ab.sh, compare_samples.py, validate.py.
  - `meta/`, `logs/`, `eval/`, `data`는 gitignore.
- `inference.py`: 학습한 화풍으로 이미지를 생성하는 스크립트(베이스 + [8-step] + 화풍 LoRA). 출력은 `outputs/`.

## 환경 제약
- Python: `/root/venvs/sn15/bin/python` (`--system-site-packages`로 시스템 torch 2.8 재사용). 시스템 pip는 PEP 668로 막혀 있다.
- GPU A40 45GiB 1장. 학습 피크 약 35GiB, 추론 약 34GiB라서 **둘을 동시에 돌릴 수 없다.**
- 컨테이너 RAM 약 50GB, CPU 7.65코어 제한(cgroup). `free`/`nproc`의 503GB/96코어는 호스트 값이다.
  - 그래서 `--enable_model_cpu_offload`, `--initialize_model_on_cpu`는 쓰지 않는다.
- 학습 명령은 `DiffSynth-Studio/` 루트 기준 상대경로를 쓴다.
  - 환경변수: `DIFFSYNTH_MODEL_BASE_PATH=<repo>/DiffSynth-Studio/models`, `DIFFSYNTH_SKIP_DOWNLOAD=true`

## 틀리기 쉬운 것
- LoRA 대상은 반드시 7개 `_mot_gen` 이름을 명시한다. 파서 기본값은 SenseNova에 매칭되지 않는다.
- 메타데이터 컬럼은 `image`, `prompt`. CSV는 QUOTE_ALL로 쓴다(캡션에 쉼표가 있음).
- `--use_gradient_checkpointing`을 명시한다(파서 기본값이 False).
- `--save_steps`는 micro-step 기준이다.
- LoRA를 이어서 학습할 때는 `--lora_checkpoint`를 쓴다(`--resume_from_checkpoint` 아님).
- 이미지 RGBA는 이미 회색 배경과 합성된 RGB다. 다시 합성하지 않는다.

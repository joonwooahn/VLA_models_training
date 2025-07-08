# VLA Models Ablation Study

완전한 factorial design을 사용한 VLA 모델 ablation study 시스템입니다.

## 실험 설계

### 요인 (Factors)
1. **모델 (Model)**: 3가지
   - GR00T
   - PI0  
   - UniVLA

2. **데이터 양 (Data Amount)**: 2가지
   - 20% (약 12 episodes)
   - 100% (전체 데이터)

3. **상태 구성 (State Configuration)**: 2가지
   - `pos_only`: joint position만 사용 (torque & velocity 제외)
   - `full_state`: position + torque + velocity 모두 사용

4. **액션 타입 (Action Type)**: 2가지
   - `single_arm`: 몸 + 오른팔 + 오른손 (32차원)
   - `bimanual`: 몸 + 양팔 + 양손 (42차원)

5. **카메라 뷰 (Camera Views)**: 2가지
   - `robot_view`: robotview만
   - `multi_view`: robotview + sideview + wrist_views

### 총 실험 수
**3 × 2 × 2 × 2 × 2 = 48개 실험**

각 모델당 16개 조건, 총 48개 조건이 자동으로 생성됩니다.

## 사용법

### 1. 모든 조건 확인
```bash
# 요약 정보 보기
python run_ablation.py --summary

# 모든 조건 리스트 보기
python run_ablation.py --list
```

### 2. 특정 조건 실행
```bash
# 특정 조건 실행 (dry-run)
python run_ablation.py --condition gr00t_20_percent_pos_only_single_arm_robot_view --dry-run

# 실제 실행
python run_ablation.py --condition gr00t_20_percent_pos_only_single_arm_robot_view
```

### 3. 모델별 모든 조건 실행
```bash
# GR00T 모델의 모든 16개 조건 실행
python run_ablation.py --model gr00t --dry-run

# PI0 모델의 모든 16개 조건 실행
python run_ablation.py --model pi0

# UniVLA 모델의 모든 16개 조건 실행
python run_ablation.py --model univla
```

### 4. 모든 조건 실행
```bash
# 모든 48개 조건 실행 (매우 오래 걸림)
python run_ablation.py --all
```

## 조건 명명 규칙

조건 이름은 다음과 같은 패턴을 따릅니다:
```
{model}_{data_amount}_{state_type}_{action_type}_{camera_type}
```

예시:
- `gr00t_20_percent_pos_only_single_arm_robot_view`
- `pi0_100_percent_full_state_bimanual_multi_view`
- `univla_20_percent_full_state_single_arm_multi_view`

## 출력 구조

각 실험의 결과는 다음과 같이 저장됩니다:

```
checkpoints/                    # GR00T 결과
├── gr00t_20_percent_pos_only_single_arm_robot_view/
├── gr00t_20_percent_pos_only_single_arm_multi_view/
└── ...

outputs/                        # PI0, UniVLA 결과
├── pi0_20_percent_pos_only_single_arm_robot_view/
├── pi0_20_percent_pos_only_single_arm_multi_view/
├── univla_20_percent_pos_only_single_arm_robot_view/
└── ...

ablation_results_*.log          # 각 실험의 로그 파일
```

## 실험 조건 예시

### GR00T (16개 조건)
1. `gr00t_20_percent_pos_only_single_arm_robot_view`
2. `gr00t_20_percent_pos_only_single_arm_multi_view`
3. `gr00t_20_percent_pos_only_bimanual_robot_view`
4. `gr00t_20_percent_pos_only_bimanual_multi_view`
5. `gr00t_20_percent_full_state_single_arm_robot_view`
6. `gr00t_20_percent_full_state_single_arm_multi_view`
7. `gr00t_20_percent_full_state_bimanual_robot_view`
8. `gr00t_20_percent_full_state_bimanual_multi_view`
9. `gr00t_100_percent_pos_only_single_arm_robot_view`
10. `gr00t_100_percent_pos_only_single_arm_multi_view`
11. `gr00t_100_percent_pos_only_bimanual_robot_view`
12. `gr00t_100_percent_pos_only_bimanual_multi_view`
13. `gr00t_100_percent_full_state_single_arm_robot_view`
14. `gr00t_100_percent_full_state_single_arm_multi_view`
15. `gr00t_100_percent_full_state_bimanual_robot_view`
16. `gr00t_100_percent_full_state_bimanual_multi_view`

PI0와 UniVLA도 동일한 패턴으로 각각 16개씩 생성됩니다.

## 시스템 장점

1. **완전한 factorial design**: 모든 조합을 체계적으로 테스트
2. **일관된 데이터 사용**: 동일한 데이터셋을 다양한 방식으로 전처리
3. **자동화된 실험 관리**: 수동 설정 없이 모든 조건 자동 생성
4. **명확한 명명 규칙**: 조건 이름만으로 실험 설정 파악 가능
5. **단계별 실행**: 개별 조건, 모델별, 또는 전체 실행 가능
6. **Dry-run 지원**: 실제 실행 전 명령어 확인 가능

## 분석 관점

이 ablation study를 통해 다음을 분석할 수 있습니다:

1. **모델 간 성능 비교**: GR00T vs PI0 vs UniVLA
2. **데이터 양의 영향**: 20% vs 100% 데이터 사용
3. **상태 정보의 중요성**: position only vs full state
4. **액션 복잡도**: single arm vs bimanual
5. **시각 정보의 효과**: single view vs multi view
6. **교차 효과**: 각 요인 간의 상호작용

총 48개의 실험을 통해 VLA 모델의 성능에 영향을 미치는 요인들을 체계적으로 분석할 수 있습니다. 
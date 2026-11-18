# 신용카드 사용자 신용도 예측

신용카드 사용자의 인적·소득·카드 정보를 바탕으로 신용도 등급(credit 0 / 1 / 2)을 확률로 예측하는 다중 분류 프로젝트

- 주 평가지표 : **log-loss** (예측 확률의 정확도)
- 보조 평가지표 : **mcaro-F1** (소수 클래스 성능)
- 최종 모델 : XGBoost·LightGBM·CatBoost soft Voting + 클래스별 임계값 조정
- 최종 성능 : log-loss 0.8086, macro-F1 0.4284

## 1. 프로젝트 개요
### 목표
사용자별 신용도 등급을 단순히 하나로 분류하는 것이 아니라, 각 등급에 속할 확률을 최대한 정확하는 것을 목표로 함

### 데이터
| 항목 | 내용 |
| --- | --- |
| 원본 | 26,457행 X 20컬럼 (train.csv) |
| 정제 후 | 24,823행 (완전 중복 1,634행 제거) |
| 고유 사용자 | 8,606명
| 타깃 | credit 0 / 1 / 2 (값이 낮을수록 신용도가 높음) |

### 타깃 분포 (원본)
| credit | 건수 | 비율 |
| --- | --- | --- |
| 0 | 3,222 | 12.2% |
| 1 | 6,267 | 23.7% |
| 2 | 16,968 | 64.1% |

credit 2가 전체의 약 64%를 차지하는 **불균형 데이터**

## 2. 파이프라인
```
01. 원본 데이터 로드
          ↓
02. 데이터 분석 (EDA)
          ↓
03. 데이터 전처리
          ↓
04. 파생 변수 추가
          ↓
05. 학습 / 평가 데이터 분리
          ↓
06. 기본 모델 학습 및 성능 평가
          ↓
07. 유효 변수 선별
          ↓
08. 선별 후 기본 모델 재평가
          ↓
09. 하이퍼파라미터 튜닝 및 임계값 조정
```
원본 데이터를 시작으로 데이터 전처리, 파생 변수 추가, 데이터 분리 후의 각 데이터를 MySQL 테이블로 저장하고, <br>모든 단계를 MySQL의 테이블에서 불러와서 실행

## 3. 데이터 분석 (EDA)
### 결측값
- `occyp_tpye`(직업)만 8,171(30.9%) 결측
- 결측자의 income_type을 확인하니 Pensioner(연금 수령자)가 4,440명, 나머지가 3,731명
- Pensioner 4,449명 중 직업이 있는 사람은 9명 뿐이므로, 은퇴로 인해 직업이 비어 있는 것으로 판단

| 처리 | 대상 |
| --- | --- |
| `Pensioner`로 채움 | income_type이 Pensioner인 결측 |
| `Unknown`으로 채움 | 그 외 결측 |

### 이상값
| 변수 | 발견 | 판단 | 처리 |
| --- | --- | --- | --- |
| `DAYS_EMPLOYED` | 365,243(약 1,000년)이 4,438건 | 전부 Pensioner → 무직 표시값 | 근속연수 0으로 변환 |
| `child_num` | 최대 19명 | 실제 대가족으로 추정 | 5명으로 클리핑 |
| `fmaily_size` | 최대 20명 | 동일 | 7명으로 클리핑 |
| `FLAG_MOBIL` | 전부 1 | 분산 0, 정보 없음 | 제거 |

### 중복 데이터와 user_id
- 모든 컬럼이 같은 완전 중복이 1,634건이므로 제거
- 같은 사람이 카드를 여러 번 신청한 것으로 보여 묶어서 user_id 생성
- 변수를 늘려 가며 검증한 끝에 6개 변수로 확정

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; user_id = gender + DAYS_BIRTH + edu_type + income_type + house_type + family_type 

- 이 기준으로 묶으니 고유 고유 사용자는 8,606명, 한 사람이 최대 30번까지 등록되어 있었음
- user_id는 이후 데이터 분리와 교차 검증에서 같은 사람이 학습과 검증에 동시에 들어가지 않도록 막는 데 사용

## 4. 데이터 전처리
| 처리 | 내용 |
| --- | --- |
| 결측 처리 | occyp_type → Pensioner / Unknown |
| 클리핑 | child_num 최대 5, family_size 최대 7 |
| 변수 제거 | FLAG_MOBIL |
| 중복 제거 | 24,823행 남음 |
| user_id 생성 | 6개 변수 조합 |
| 음수 값 변환 | DAYS_BIRTH → age(나이), DAYS_EMPLOYED → employment_years(근속연수, 무직은 0), begin_month → card_months(카드 보유 개월) |
| 스케일링 | Logisitc Regression |
| 원-핫 인코딩 | XGBoost / LightGBM |

## 5. 파생변수
| 변수 | 계산 | 의미 |
| --- | --- | --- |
| `log_income ` | log(1 + income_total) | 소득의 왜도 완화
| `income_to_age_ratio` | income_total | 나이 대비 소득 수준 |
| `income_per_family` | income_total / family_size | 가족 1인당 소득 |

## 6. 학습 / 평가 데이터 분리
- `GroupShuffleSplit`으로 user_id 단위 80(학습) / 20(평가) 분리
- train 19,905행 / test 4,918, 겹치는 사용자 0명
- test 데이터는 최종 평가 전까지 미사용

## 7. 기본 모델 성능 평가
### 평가 방식
- `StratifiedGroupKFold`(5-fold) : 클래스 비율을 유지하면서 user_id가 fold 사이에 겹치지 않도록 분할
- 모든 지표는 5-fold OOF 예측으로 계산

### 평가지표 선정
| 지표 | 역할 | 선정 이유 |
| --- | --- | --- |
| log-loss | 주 지표 | 목표가 등급별 확률 산출이므로, 확률 자체가 얼마나 정확한지를 평가 |
| macro-F1 | 보조 지표 | 불균형 데이터에서 소수 클래스(credit 0)도 동일한 비중으로 반영 |
| accuracy | 참고 | 모든 사람을 credit 2로 찍어도 0.64가 나오므로 단독 판단에 부적합 |

### 결과 (log-loss)
| 모델 | 클래스 가중치 O | 클래스 가중치 X|
| --- | --- | --- |
| Naive 전부 credit 2) | 12.9236 | - |
| Logistic Regression | 1.0780 | 0.8675 |
| XGBoost | 1.0668 | 1.0850 |
| LightGBM | 0.9546 | 0.9286 |
| CatBoost | 0.9714 | **0.8422** |

- 클래스 가중치(balanced)를 주면 소수 클래스 recall은 오르지만, log-loss가 대부분 나빠짐
- log-loss를 주 지표로 정했기 때문에, 이후 단계는 가중치 없이 진행

## 8. 유효 변수 선별
| 지표 | 내용 |
| --- | --- |
| `log_income` | SHAP 0, Permutation 0. 트리 모델은 순서만 보므로 income_total과 정보가 중복 |
| `family_size` | SHAP 0.0099, IV 0.0001, VIF 12.2 (child_num과 다중공선성)
| `work_phone` | IV 0, Permutation 0.0044 |
| `email` | SHAP 0.0078, IV 0, Permutation 0.0027 |

Permutation이나 IV 하나에서만 낮게 나온 변수(gender, car 등)는 다른 지표에서 기여가 확인되어 유지하고,<br> 여러 지표에서 공통으로 낮게 나온 변수만 제거

## 9. 선별 후 기본 모델 재학습 및 평가
| 모델 | 선별 전 | 선별 후 |
| --- | --- | --- |
| Logistic Regression | 0.8675 | 0.8670 |
| XGBoost | 1.0850 | 1.0817 |
| LightGBM | 0.9286 | 0.9283 |
| CatBoost | 0.8422 | 0.8388 |

변수 4개를 줄였는데도 네 모델 모두 그 전의 결과와 차이가 거의 없어 불필요한 변수였다는 것을 확인

## 10. 하이퍼파라미터 튜닝 · 앙상블 · 임계값 조정
### 튜닝 설정
- Optuna TPE Sampler + MedianPruner, 모델별 100 trials
- 목적 함수 : 5-fold 평균 log-loss 최소화
- Early stopping(patience 50)으로 fold별 최적 트리 수를 찾고, 그 평균을 최종 트리 수로 사용

### 튜닝 결과 (OOF)
| 모델 | 기본 모델 log-loss | 튜닝 후 log-loss | macro-F1 | accuracy |
| --- | --- | --- | --- | --- |
| XGBoost | 1.0817 | **0.8074** | 0.3829 | 0.6914 |
| LightGBM | 0.9283 | 0.8114 | 0.3859 | 0.6912 |
| CatBoost | 0.8388 | 0.8094 | 0.3921 | 0.6912 |

- 세 모델 모두 log-loss가 0.81 안팎으로 수렴
- 세 모델 모두 credit 0의 recall이 0. 확률이 가장 높은 클래스를 고르는 방식으로는 소수 클래스가 한 번도 선택되지 않음

### 앙상블 (Soft Voting)
|     | log-loss | macro-F1 | accuracy |
| --- | --- | --- | --- |
| Voting Ensemble | 0.8086 | 0.3869 | 0.6911

세 모델의 예측이 서로 비슷해서 더 좋은 성능은 얻지 못함

### 임계값 조정
앙상블 확률에 클래스별 가중치를 곱한 뒤 가장 큰 클래스를 고르도록 분류 기준 조정. 가중치는 Optuna로 macro-F1이 최대가 되도록 탐색


|    | 조정 전 | 조정 후 |
| --- | --- | --- |
| log-loss | 0.8086 | 0.8086|
| macro-F1 | 0.3869 | **0.4284** |
| accuracy | 0.6911 | 0.6187 |
| credit 0 recall | 0.0000 | **0.1779** |
| credit 1 recall | 0.2288 | 0.2444 |
| credit 2 recall | 0.9932 | 0.8405 |

- credit 0을 처음으로 예측하기 시작했고, macro-F1이 0.387에서 0.428로 오름
- 그 대신 credit 2의 recall과 accuracy는 낮아짐
- 확률 자체는 바꾸지 않고 분류 기준만 조정했기 때문에 log-loss는 그대로임

### 최종 모델
| 항목 | 내용 |
| --- | --- |
| 구성 | XGBoost + LightGMB + CatBoost Soft Voting |
| 분류 기준 | 클래스별 가중치 [4.242, 1.830, 0.810] |
| log-loss | 0.8086 |
| macro-F1 | 0.4284

## 11. 기술 스택
| 구분 | 사용 도구 |
| --- | --- |
| 언어 / 환경 | Python, Jupyter Notebook, Anaconda |
| 데이터베이스 | MySQL, SQLAlchemy, PyMYSQL |
| 데이터 처리 | Pnadas, Numpy |
| 통계 검정 | Scipy, statsmodels |
| 모델링 | Scikit-learn, XGBoost, LightGBM, CatBoost |
| 튜닝 | Optuna |
| 해석 | SHAP |
| 시각화 | Matplotlin, Seaborn |

## 12. 폴더 구조
```
├── data/
│     └─ train/
├── models
│     ├─ xgb_tuned_noweight.pkl
│     ├─ lgb_tuned_noweight.pkl
│     ├─ cat_tuned_noweight.pkl
│     ├─ ensemble_noweight.pkl
└── notebooks      
      ├─ 01_load_raw.ipynb
      ├─ 02_eda.ipynb
      ├─ 03_prepare.ipynb
      ├─ 04_features.ipynb
      ├─ 05_split.ipynb
      ├─ 06_baseline_models.ipynb
      ├─ 07_feature_validation.ipynb
      ├─ 08_baseline_v2.ipynb
      └─ 09_tuning.ipynb
```

## 13. 회고
### 배운 점
- Feature를 다양한 근거로 보는 경험 : 이전엔 상관계수 및 다중고선성 위주로 변수의 선택 여부를 고민했다면 이번엔 5개의 지표로 변수 선택을 고려해볼 수 있었음
### 아쉬운 점
- 데이터의 구조적 한계 : 근속 기간은 현 직장뿐이고 이직 횟수나 총 경력이 없음. 개인정보라 외부 데이터 결합 불가능
- 데이터 탐색의 깊이 부족 : 초기엔 '개인 식별 번호'를 변하기 어려운 3개의 변수 기준으로 묶었지만, 마지막 점검 때 좀 더 깊이 들여다 본 뒤 6개로 재 설계
### 개선 방향
- 데이터에 대한 충분한 공부와 이해 : EDA에 충분히 시간을 투자하고 데이터의 구조와 의문점은 깊이 파고들어야 함



































    

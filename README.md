# DiseaseConqueror
머신러닝과 국민건강영양조사 데이터 기반의 만성질환 질병 정복자

데이터 다운로드 링크: https://dataon.kisti.re.kr/search/view.do?mode=view&svcId=c3aaeefa557772ed8c57515a0793ffab

# DiseaseConqueror.py
"""<br/>
  &nbsp;&nbsp;args<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;datadir: 데이터 디렉토리<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year: 연도<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target: 분석하고자 하는 질병<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;models: 사용하고자 하는 머신러닝 모델<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;one_hot_encoding: 웟-핫 인코딩 여부<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;n_splits: k-fold<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;test_size: test size<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;random_state: random state<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;save: 저장 여부<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;metrics: 머신러닝 평가 지표<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;n1: factor_extraction 함수의 n<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;n2: confusion_matrix 함수의 n<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;visualization: 시각화 여부<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prePath: 저장 경로<br/>
"""<br/>

터미널 사용 방법<br/>
>python DiseaseConqueror.py -datadir './data' -year 2022 -target '당뇨병' -models RandomForest AdaBoost -one_hoe_encoding True -n_splits 5 -test_size 0.33 -random_state 42 -save True -metrics Accuracy Precision -n1 40 -n2 20 -visualization True -prePath './data'<br/>

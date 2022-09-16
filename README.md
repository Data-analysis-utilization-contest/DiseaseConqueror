# DiseaseConqueror
머신러닝과 국민건강영양조사 데이터 기반의 만성질환 질병 정복자

데이터 다운로드 링크: https://dataon.kisti.re.kr/search/view.do?mode=view&svcId=c3aaeefa557772ed8c57515a0793ffab

터미널 사용 방법: 
python DiseaseConqueror.py -datadir -year -target -models -one_hot_encoding -n_splits -test_size -random_state -save -metrics -n1 -n2 -visualization -prePath<br/>
"""
  <br/>args<br/>
    <br/>datadir: 데이터 디렉토리<br/>
    <br/>year: 연도<br/>
    <br/>target: 분석하고자 하는 질병<br/>
    <br/>models: 사용하고자 하는 머신러닝 모델<br/>
    <br/>one_hot_encoding: 웟-핫 인코딩 여부<br/>
    <br/>n_splits: k-fold<br/>
    <br/>test_size: test size<br/>
    <br/>random_state: random state<br/>
    <br/>save: 저장 여부<br/>
    <br/>metrics: 머신러닝 평가 지표<br/>
    <br/>n1: <br/>
    <br/>n2: <br/>
    <br/>visualization: 시각화 여부<br/>
    <br/>prePath: 저장 경로<br/>
<br/>"""<br/>
<br/>예시 코드<br/>
<br/>python DiseaseConqueror.py -datadir './data' -year 2022 -targt '당뇨병' -models RandomForest AdaBoost -one_hoe_encoding True -n_splits 5 -test_size 0.33 -random_state 42 -save True -metrics Accuracy Precision -n1 40 -n2 20 -visualization True -prePath './data'<br/>

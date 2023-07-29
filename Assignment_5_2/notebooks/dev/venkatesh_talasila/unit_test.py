from sklearn.utils.estimator_checks import check_estimator
from custom_transformer import CustomTransformer
def test_custom_transformer():
    check_estimator(CustomTransformer())
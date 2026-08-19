from fruit_grader.inference import prediction_from_values


def test_fresh_prediction_uses_chinese_fresh_message():
    prediction = prediction_from_values({0: "fresh", 1: "spoiled"}, 0, 0.875)

    assert prediction.label == "fresh"
    assert prediction.confidence == 0.875
    assert prediction.message == "判别结果：新鲜"


def test_unknown_prediction_is_not_mislabeled():
    prediction = prediction_from_values({0: "unknown"}, 0, 0.5)

    assert prediction.message == "判别结果：未知类别（unknown）"

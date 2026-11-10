from __future__ import annotations

from app.training import data
from app.training.pipeline import check_for_training_data
from tests.training_fixtures import make_dataset, write_good_file


def test_split_uses_last_twenty_percent_as_test_set() -> None:
    dataset = make_dataset(100)

    train, test = data.split_train_test(dataset)

    assert len(train) == 80
    assert len(test) == 20
    assert train["instant"].max() == 80
    assert test["instant"].min() == 81


def test_split_matches_the_baseline_split_size() -> None:
    dataset = make_dataset(17379)

    train, test = data.split_train_test(dataset)

    assert len(train) == 13903
    assert len(test) == 3476


def test_label_new_rows_looks_up_cnt_by_instant(tmp_path) -> None:
    dataset = make_dataset(100)
    path = write_good_file(tmp_path, "a.csv", dataset, [5, 6])

    new_rows = data.read_good_files(tmp_path, [path.name])
    labeled = data.label_new_rows(new_rows, dataset)

    assert list(labeled["instant"]) == [5, 6]
    assert list(labeled["cnt"]) == [14, 15]


def test_label_new_rows_handles_no_files(tmp_path) -> None:
    dataset = make_dataset(100)

    new_rows = data.read_good_files(tmp_path, [])
    labeled = data.label_new_rows(new_rows, dataset)

    assert labeled.empty


def test_training_set_never_contains_test_rows(tmp_path) -> None:
    dataset = make_dataset(100)
    write_good_file(tmp_path, "a.csv", dataset, [3, 4, 85, 90])

    train, test, info = data.build_training_set(dataset, tmp_path, ["a.csv"])

    assert train["instant"].max() == 80
    assert set(train["instant"]).isdisjoint(set(test["instant"]))
    assert info["new_rows_read"] == 4
    assert info["new_rows_in_test_period"] == 2


def test_training_set_removes_copies_of_original_rows(tmp_path) -> None:
    dataset = make_dataset(100)
    write_good_file(tmp_path, "a.csv", dataset, [3, 4, 5])

    train, _, info = data.build_training_set(dataset, tmp_path, ["a.csv"])

    assert len(train) == 80
    assert train["instant"].is_unique
    assert info["train_rows"] == 80


def test_find_good_files_ignores_files_already_trained(tmp_path) -> None:
    dataset = make_dataset(100)
    good_dir = tmp_path / "good"
    state_file = tmp_path / "state" / "last_training.json"
    write_good_file(good_dir, "a.csv", dataset, [1])
    write_good_file(good_dir, "b.csv", dataset, [2])
    data.write_trained_files(state_file, {"a.csv"})

    all_files, new_files = data.find_good_files(good_dir, state_file)

    assert all_files == ["a.csv", "b.csv"]
    assert new_files == ["b.csv"]


def make_config(tmp_path, minimum: int) -> dict:
    return {
        "good_data_dir": str(tmp_path / "good"),
        "state_file": str(tmp_path / "state.json"),
        "min_new_rows": minimum,
    }


def test_first_run_trains_even_without_new_data(tmp_path) -> None:
    result = check_for_training_data(make_config(tmp_path, 100), False)

    assert result["should_train"] is True


def test_skips_when_not_enough_new_rows(tmp_path) -> None:
    dataset = make_dataset(100)
    write_good_file(tmp_path / "good", "a.csv", dataset, [1, 2, 3])

    result = check_for_training_data(make_config(tmp_path, 10), True)

    assert result["should_train"] is False
    assert result["new_rows"] == 3


def test_trains_when_enough_new_rows(tmp_path) -> None:
    dataset = make_dataset(100)
    write_good_file(tmp_path / "good", "a.csv", dataset, list(range(1, 11)))

    result = check_for_training_data(make_config(tmp_path, 10), True)

    assert result["should_train"] is True
    assert result["good_files"] == ["a.csv"]

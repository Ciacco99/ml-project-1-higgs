import numpy as np

from implementations import *
from helpers import *
from plots import *


def k_fold_test(y, tX, k_fold, lambda_, verbatim=True):
    avg_accuracy_tr = 0
    avg_accuracy_te = 0
    k_indices = build_k_indices(tX.shape[0], k_fold)
    for k in range(k_fold):
        indices_te = k_indices[k]
        y_te, tX_te = y[indices_te], tX[indices_te]
        indices_tr = np.delete(k_indices, k, axis=0).flatten()
        y_tr, tX_tr = y[indices_tr], tX[indices_tr]

        initial_w = np.zeros(tX_tr.shape[1])
        w, loss = ridge_regression(y_tr, tX_tr, lambda_)

        y_pred_te = predict_labels(w, tX_te)
        y_pred_tr = predict_labels(w, tX_tr)

        accuracy_te = accuracy(y_te, y_pred_te)
        accuracy_tr = accuracy(y_tr, y_pred_tr)
        avg_accuracy_tr += accuracy_tr / k_fold
        avg_accuracy_te += accuracy_te / k_fold

    if verbatim:
        print("Average {}-fold train accuracy: {}".format(k_fold, avg_accuracy_tr))
        print("Average {}-fold test accuracy: {}".format(k_fold, avg_accuracy_te))
    return avg_accuracy_tr, avg_accuracy_te


def degree_search(y_tr, tX_tr):
    best_accuracy = 0
    best_degree = 0
    for degree in range(6, 16):
        print("/// Degree: {}".format(degree))
        tX = feature_expansion(tX_tr, degree=degree)
        y, tX = augment(y_tr, tX, noise=0.025)
        y, tX = augment(y, tX, noise=0.01)

        _, accuracy = k_fold_test(y, tX, 3)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_degree = degree

    print(">>> Best degree: {}".format(best_degree))


def lambda_search(y_tr, tX_tr):
    lambdas = np.logspace(-7, -1.5, num=15)
    accuracies_tr = []
    accuracies_te = []
    for lambda_ in lambdas:
        # print("/// Lambda: {}".format(lambda_))

        acc_tr, acc_te = k_fold_test(y_tr, tX_tr, 3, lambda_, verbatim=False)
        accuracies_tr.append(acc_tr)
        accuracies_te.append(acc_te)

    return lambdas, np.array(accuracies_tr), np.array(accuracies_te)


def overfitting_search(y_tr, tX_tr, file_prefix, threshold=0.997):
    for i, noise in enumerate([-1, 0.025, 0.01, 0.005, 0.0025, 0.001]):
        print("///// Augmented {} times".format(i))
        if noise != -1:
            y_tr, tX_tr = augment(y_tr, tX_tr, noise=noise)
        base_acc_tr, base_acc_te = k_fold_test(y_tr, tX_tr, 3, 0)
        base_ratio = base_acc_te / base_acc_tr
        print("Overfitting ratio: {}".format(base_ratio))

        lambdas, acc_tr, acc_te = lambda_search(y_tr, tX_tr)

        best_index = np.argmax(acc_te)
        best_acc_ratio = acc_te[best_index] / acc_tr[best_index]
        print("// Best lambda: {}".format(lambdas[best_index]))
        print("Test accuracy: {}".format(acc_te[best_index]))
        print("Accuracy ratio: {}".format(best_acc_ratio))

        file_name = "{}_aug_{}x".format(file_prefix, i)
        plot_lambda_search(lambdas, acc_tr, acc_te, "plots/" + file_name + ".png")
        data = np.stack((lambdas, acc_tr, acc_te), axis=-1)
        np.savetxt(
            "out/" + file_name + ".csv",
            data,
            delimiter=",",
            header="Lambda, Train accuracy, Test accuracy",
        )
        if best_acc_ratio > threshold:  # If not overfitting by much, break
            break


def main():
    # Select what tasks to perform
    LAMBDA_SEARCH = False
    K_FOLD_TEST = False
    OUTPUT_SUBMISSION = True

    # Optimal degrees found using hyperparameter search
    # Usually picked maximal degree, above which things stopped
    # Being stable
    degrees = {
        (True, 0): 10,
        (True, 1): 10,
        (True, 2): 10,
        (True, 3): 10,
        (False, 0): 7,
        (False, 1): 10,
        (False, 2): 10,
        (False, 3): 10,
    }

    augment_times = {
        (True, 0): 0,
        (True, 1): 1,
        (True, 2): 2,
        (True, 3): 3,
        (False, 0): 0,
        (False, 1): 2,
        (False, 2): 4,
        (False, 3): 4,
    }

    lambdas = {
        (True, 0): 0.0004,
        (True, 1): 0.001,
        (True, 2): 0.00005,
        (True, 3): 0.0006,
        (False, 0): 0.000003,
        (False, 1): 0.0,
        (False, 2): 0.0,
        (False, 3): 0.000001,
    }

    # Setting seed at beginning of program ensures we
    # get the same results every time even with randomness
    np.random.seed(1)

    # Load column names from CSV
    cols = load_csv_names("data/train.csv")

    # Each column with name key is NaN (-999.0) if jet_num < value
    min_jet_num = {
        "DER_deltaeta_jet_jet": 2,
        "DER_mass_jet_jet": 2,
        "DER_prodeta_jet_jet": 2,
        "DER_lep_eta_centrality": 2,
        "PRI_jet_leading_pt": 1,
        "PRI_jet_leading_eta": 1,
        "PRI_jet_leading_phi": 1,
        "PRI_jet_subleading_pt": 2,
        "PRI_jet_subleading_eta": 2,
        "PRI_jet_subleading_phi": 2,
        "PRI_jet_all_pt": 1,
    }
    # Array containing for each jet_num an np.array with boolean values
    # True if column is NaN for a given jet_num
    NaN_by_jet_num = [
        np.array([min_jet_num.get(var, 0) > jet_num for var, index in cols.items()])
        for jet_num in range(4)
    ]

    # Load train and test data from CSV
    y_train, tX_train, ids_train = load_csv_data("data/train.csv")
    if OUTPUT_SUBMISSION:
        _, tX_test, ids_test = load_csv_data("data/test.csv", sub_sample=False)
    else:
        _, tX_test, ids_test = (
            np.zeros(1),
            np.zeros((1, tX_train.shape[1])),
            np.zeros(1),
        )

    # Fetch column indices for columns of particular interest
    PRI_jet_num = cols["PRI_jet_num"]
    DER_mass_MMC = cols["DER_mass_MMC"]

    # Split data by jet number
    jet_split_tr = [tX_train[:, PRI_jet_num] == i for i in range(4)]
    jet_split_te = [tX_test[:, PRI_jet_num] == i for i in range(4)]

    # Split data by whether DER_mass_MMC is defined or not
    mass_MMC_NaN_tr = tX_train[:, DER_mass_MMC] == -999.0
    mass_MMC_NaN_te = tX_test[:, DER_mass_MMC] == -999.0

    # Vector where we'll write the predictions
    y_prediction = np.zeros(tX_test.shape[0])

    # Main loop
    # - iterate over all subsamples split by jet number
    # and whether DER_mass_MMC is defined or not
    # - We get 8 different subsamples for which we
    # perform linear regression separately
    for with_mass_MMC in [True, False]:
        for jet_num in [0, 1, 2, 3]:
            print("////////////////////////////////")
            print("Jet number: {}, with mass_MMC: {}".format(jet_num, with_mass_MMC))

            # Select rows that correspond to the given jet number
            # and that include/don't include DER_mass_MMC
            rows_tr = jet_split_tr[jet_num]
            rows_te = jet_split_te[jet_num]
            if with_mass_MMC:
                rows_tr = rows_tr & ~mass_MMC_NaN_tr
                rows_te = rows_te & ~mass_MMC_NaN_te
            else:
                rows_tr = rows_tr & mass_MMC_NaN_tr
                rows_te = rows_te & mass_MMC_NaN_te

            # Remove columns that are not defined for jet num
            columns = ~NaN_by_jet_num[jet_num]

            # remove column PRI_jet_num
            # (the column doesn't carry any info since we
            # split the data)
            columns[PRI_jet_num] = False
            # Same with DER_mass_MMC if it's undefined
            if ~with_mass_MMC:
                columns[DER_mass_MMC] = False

            # Extract relevant training data
            y_tr = y_train[rows_tr]
            tX_tr = tX_train[rows_tr, :][:, columns]

            # Remove outliers
            y_tr, tX_tr = remove_outliers(y_tr, tX_tr)
            # Standardize the data
            tX_tr = standardize(tX_tr)

            # Perform feature expansion with polynomials of
            # degree corresponding to the dictionary we defined above
            poly_degree = degrees[(with_mass_MMC, jet_num)]
            tX_tr = feature_expansion(tX_tr, degree=poly_degree)

            if LAMBDA_SEARCH:
                overfitting_search(
                    y_tr, tX_tr, "lambda_search_({}, {})".format(with_mass_MMC, jet_num)
                )

            if not (OUTPUT_SUBMISSION or K_FOLD_TEST):
                continue

            noise_list = [0.025, 0.01, 0.005, 0.0025, 0.001]
            n_augment = augment_times[(with_mass_MMC, jet_num)]
            for noise in noise_list[:n_augment]:
                y_tr, tX_tr = augment(y_tr, tX_tr, noise=noise)

            lambda_ = lambdas[(with_mass_MMC, jet_num)]

            if K_FOLD_TEST:
                k_fold_test(y_tr, tX_tr, 3, lambda_)

            if not OUTPUT_SUBMISSION:
                continue

            # Calculate best weights for given case
            w, loss = ridge_regression(y_tr, tX_tr, lambda_)

            # Format the test data to fit the same model
            tX_te = tX_test[rows_te, :][:, columns]
            tX_te = standardize(tX_te)
            tX_te = feature_expansion(tX_te, degree=poly_degree)

            # Apply weights to test data to obtain predictions
            y_te = predict_labels(w, tX_te)
            # Store them in the final results vector
            y_prediction[rows_te] = y_te

    # Generate CSV with results
    if OUTPUT_SUBMISSION:
        create_csv_submission(ids_test, y_prediction, "ridge_spliced.csv")


if __name__ == "__main__":
    main()

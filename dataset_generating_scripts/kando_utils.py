from datetime import datetime, timezone

# from kando.kando_client import KandoClient


def convert_date_to_timestamp(date):
    """Convert date in format DD/MM/YYYY to timestamp"""

    day, month, year = list(map(int, date.split("/")))
    date_timestamp = datetime.timestamp(
        datetime.strptime(date, "%d/%m/%Y").replace(tzinfo=timezone.utc)
    )  # set UTC timezone
    return date_timestamp


def retrieve_data(POINT_IDS, secret, start_date, end_date, verbose=0):

    data_list = []
    failure_list = []

    start_date_ts = convert_date_to_timestamp(start_date)
    end_date_ts = convert_date_to_timestamp(end_date)
    client = KandoClient(
        host=secret["HOST"],
        auth_key=secret["AUTH_KEY"],
        auth_secret=secret["AUTH_SECRET"],
    )
    for POINT_ID in POINT_IDS:
        try:
            data_list.append(
                client.get_all(point_id=POINT_ID, start=start_date_ts, end=end_date_ts)
            )
            if verbose == 1:
                print(f"SUCCESS poind_id = {POINT_ID}")
        except:
            if verbose == 1:
                print(f"FAILURE point_id: {POINT_ID}")
            failure_list.append(POINT_ID)

    if verbose == 1:
        print(
            f"Successfuly retrieved data dictionary from {len(data_list)}/{len(POINT_IDS)} points."
        )
        if failure_list:
            print(f"FAILURE point_ids: {failure_list}")

    return data_list


def verify_data_not_empty(data_list, start_date, end_date, POINT_IDS):
    for index, data in enumerate(data_list):
        assert data[
            "samplings"
        ], f"No data between {start_date} and {end_date} for {POINT_IDS[index]}"
    print("Data samples not empty.")


def convert_samples_to_dataframe(data_list, POINT_IDS, verbose=0):

    df_points = {}
    features = ["COD", "EC", "PH", "TEMPERATURE"]
    for index, data in enumerate(data_list):
        df_temp = (
            pd.DataFrame.from_dict(data["samplings"], orient="index")
            .reset_index()
            .drop(columns=["index"])
            .set_index(keys="DateTime")
        )
        df_temp.index = pd.to_datetime(
            df_temp.index, unit="s"
        )  # Convert index from Unix/Epoch time to Readable date format
        df_temp = df_temp[features]
        df_points[POINT_IDS[index]] = df_temp
    print("DataFrames created.")

    return df_points


def remove_faulty_records(df, faulty_dict, point_id=883, verbose=0):

    df_clean = df.copy()
    for _, location in faulty_dict.items():
        if "cod_pointid" in location.keys():
            if location["cod_pointid"] == point_id:
                for corr_dates in location["corrupt_data_ind"]:
                    start_date, end_date = corr_dates[0], corr_dates[1]
                    mask = (df_clean.index < start_date) | (df_clean.index > end_date)
                    df_clean = df_clean.loc[mask]
    if verbose == 1:
        percentage_removed = (1 - df_clean.shape[0] / df.shape[0]) * 100
        print(
            f"Removed {df.shape[0]-df_clean.shape[0]} records, which is {percentage_removed:.2f}% data"
        )

    return df_clean

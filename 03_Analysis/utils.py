import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import polygonize

METRICS_COLUMNS = [
    "creator_1",
    "creator_2",
    "IoU",
    "F1",
    "df1_area_recall",
    "df2_area_recall",
    "df1_area_precision",
    "df2_area_precision",
    "df1_number_recall",
    "df2_number_recall",
    "df1_number_precision",
    "df2_number_precision",
    "intersection_area",
    "union_area",
    "df1_unique_area",
    "df2_unique_area",
    "df1_number",
    "df1_intersection_number",
    "df1_unique_number",
    "df2_number",
    "df2_intersection_number",
    "df2_unique_number",
]


def normalize(array, p_low=None, p_high=None):
    """
    Normalize an input array to the range [0, 1] based on specified or default percentiles.

    This function normalizes the input array by scaling its values to the range [0, 1].
    It can use either the minimum and maximum values of the array or specified percentiles
    as the scaling bounds.

    Parameters:
    -----------
    array : numpy.ndarray
        The input array to be normalized.
    p_low : float, optional
        The lower percentile value to use for scaling. If None, the minimum value of the array is used.
    p_high : float, optional
        The higher percentile value to use for scaling. If None, the maximum value of the array is used.

    Returns:
    --------
    numpy.ndarray
        The normalized array with values clipped to the range [0, 1].

    Notes:
    ------
    - If p_low and p_high are provided, they should be in the range [0, 100].
    - The function uses numpy's percentile function when p_low or p_high are specified.
    - Values below p_low will be set to 0, and values above p_high will be set to 1 in the output.
    """
    # make per band
    if p_low:
        # array_min = np.percentile(im, p_low)
        array_min = np.percentile(array, p_low)
    else:
        array_min = array.min()
    if p_high:
        # array_max = np.percentile(im, p_high)
        array_max = np.percentile(array, p_high)
    else:
        array_max = array.max()

    return np.clip((array - array_min) / (array_max - array_min), 0, 1)


def calculate_overlap(df1, df2, print_output=False, name_field="creator_id"):
    """
    Calculate various overlap metrics between two geometric datasets.

    This function computes intersection, union, and difference areas, as well as
    IoU (Intersection over Union), recall, precision, and F1 score for two input
    geometric datasets.

    Parameters:
    -----------
    df1 : geopandas.GeoDataFrame
        The first geometric dataset.
    df2 : geopandas.GeoDataFrame
        The second geometric dataset to compare with df1.
    print_output : bool, optional (default=False)
        If True, prints the calculated metrics to the console.
    name_field : str, optional (default='creator_id')
        The field name in the dataframes to use for identifying the creators.

    Returns:
    --------
    pandas.Series
        A series containing the following metrics:
        - intersection_area: Area of intersection between df1 and df2
        - union_area: Area of union between df1 and df2
        - df1_unique_area: Area unique to df1
        - df2_unique_area: Area unique to df2
        - IoU: Intersection over Union
        - df1_area_recall: Recall for df1
        - df2_area_recall: Recall for df2
        - df1_area_precision: Precision for df1
        - df2_area_precision: Precision for df2
        - F1: F1 score
        - creator_1: Identifier for the creator of df1
        - creator_2: Identifier for the creator of df2

    Notes:
    ------
    - The function assumes that df1 and df2 are GeoDataFrames with valid geometry columns.
    - The unary_union of each dataframe is used for calculations, combining all geometries.
    - F1 score is calculated as the harmonic mean of the recalls if possible, else 0.
    - If print_output is True, detailed metrics are printed to the console.
    """
    df1u = df1.unary_union
    df2u = df2.unary_union

    intersection = df1u.intersection(df2u)
    union = df1u.union(df2u)
    diff1 = df1u.difference(intersection)
    diff2 = df2u.difference(intersection)
    iou = intersection.area / union.area
    union1 = intersection.area + diff1.area
    union2 = intersection.area + diff2.area
    df1_area_recall = intersection.area / union1
    df2_area_recall = intersection.area / union2
    df1_area_precision = union1 / union.area
    df2_area_precision = union2 / union.area

    f1_lower = df1_area_recall + df2_area_recall
    if not f1_lower == 0:
        f1 = 2 * ((df1_area_recall * df2_area_recall) / f1_lower)
    else:
        f1 = 0

    # create series with values
    df_metrics = pd.Series(
        data=[
            intersection.area,
            union.area,
            diff1.area,
            diff2.area,
            iou,
            df1_area_recall,
            df2_area_recall,
            df1_area_precision,
            df2_area_precision,
            f1,
            df1.iloc[0][name_field],
            df2.iloc[0][name_field],
        ],
        index=[
            "intersection_area",
            "union_area",
            "df1_unique_area",
            "df2_unique_area",
            "IoU",
            "df1_area_recall",
            "df2_area_recall",
            "df1_area_precision",
            "df2_area_precision",
            "F1",
            "creator_1",
            "creator_2",
        ],
    )
    if print_output:
        print("Intersection:", intersection.area)
        print("Union:", union.area)
        print("Diff1:", diff1.area)
        print("Diff2:", diff2.area)
        print("IoU:", iou)
        print("df1_area_recall:", df1_area_recall)
        print("df2_area_recall:", df2_area_recall)
        print(df_metrics)
        print()
    return df_metrics


def create_iou_matrix_joined(df, half=True, drop_same=False, name_field="creator_id"):
    """
    Create an IoU (Intersection over Union) matrix and calculate various metrics for pairwise comparisons of geometries in a dataframe.

    This function performs pairwise comparisons between all geometries in the input dataframe,
    calculating IoU and other metrics. It can optionally compute only half of the matrix and
    exclude self-comparisons.

    Parameters:
    -----------
    df : geopandas.GeoDataFrame
        The input dataframe containing geometries to be compared.
    half : bool, optional (default=True)
        If True, computes only the upper triangular part of the matrix, assuming symmetry.
    drop_same : bool, optional (default=False)
        If True, excludes self-comparisons (i.e., comparing a geometry with itself).
    name_field : str, optional (default='creator_id')
        The field name in the dataframe to use for identifying the creators of each geometry.

    Returns:
    --------
    tuple
        A tuple containing two elements:
        1. numpy.ndarray:
           The IoU matrix where each cell [i,j] represents the IoU between geometry i and j.
        2. pandas.DataFrame:
           A dataframe containing various metrics for each pairwise comparison, including:
           - Area-based metrics: intersection area, union area, unique areas, IoU, recall, precision, F1 score
           - Count-based metrics: intersection counts, unique counts, recall, precision
           The exact columns are defined by the METRICS_COLUMNS constant.

    Notes:
    ------
    - The function uses `calculate_overlap` and `get_feature_overlap_numbers` (not shown) to compute metrics.
    - If half=True, only the upper triangular part of the matrix is computed, which is more efficient for symmetric comparisons.
    - The resulting dataframe includes both area-based and count-based metrics for each comparison.
    - The IoU matrix is symmetric, with IoU values ranging from 0 to 1.
    - The function assumes that METRICS_COLUMNS is a predefined constant listing the desired output columns.

    Example:
    --------
    >>> iou_matrix, metrics_df = create_iou_matrix_joined(gdf)
    >>> print(iou_matrix)
    >>> print(metrics_df.head())
    """
    metrics = []
    length = len(df)
    iou_array = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if drop_same and i == j:
                continue
            if half and i > j:
                continue
            df1 = df.iloc[i : i + 1]
            df2 = df.iloc[j : j + 1]
            s_count = get_feature_overlap_numbers(df1, df2)
            s_quant = calculate_overlap(df1, df2, name_field=name_field)
            s = pd.concat([s_quant, s_count])
            metrics.append(s)
            iou_array[j, i] = s.IoU
    df_metrics = pd.concat(metrics, axis=1).T
    # calculate object number metrics
    df_metrics["df1_number_recall"] = (
        df_metrics["df1_intersection_number"] / df_metrics["df1_number"]
    )
    df_metrics["df2_number_recall"] = (
        df_metrics["df2_intersection_number"] / df_metrics["df2_number"]
    )
    df_metrics["df1_number_precision"] = df_metrics["df1_intersection_number"] / (
        df_metrics["df2_unique_number"] + df_metrics["df1_intersection_number"]
    )
    df_metrics["df2_number_precision"] = df_metrics["df2_intersection_number"] / (
        df_metrics["df1_unique_number"] + df_metrics["df2_intersection_number"]
    )
    # return and reorder columns
    return iou_array, df_metrics[METRICS_COLUMNS]


def get_feature_overlap_numbers(gdf_left, gdf_right):
    """
    This function calculates the number of overlapping and individual features between two GeoDataFrames or GeoSeries.

    Parameters:
    gdf_left (geopandas.GeoDataFrame or geopandas.GeoSeries): The first GeoDataFrame or GeoSeries.
    gdf_right (geopandas.GeoDataFrame or geopandas.GeoSeries): The second GeoDataFrame or GeoSeries.

    Returns:
    pandas.Series: A series containing the number of features in `gdf_left`, `gdf_right`, the number of overlapping features, and the number of individual features in both `gdf_left` and `gdf_right`.

    The returned series has the following indices:
    - 'df1_number': The number of features in `gdf_left`.
    - 'df2_number': The number of features in `gdf_right`.
    - 'intersection_number': The number of overlapping features between `gdf_left` and `gdf_right`.
    - 'df1_diff_number': The number of features only in `gdf_left`.
    - 'df2_diff_number': The number of features only in `gdf_right`.
    """
    # transform GeoSeries to GeoDataFrame
    if isinstance(gdf_left, pd.Series):
        gdf_left = gpd.GeoDataFrame(gdf_left).T
    if isinstance(gdf_right, pd.Series):
        gdf_right = gpd.GeoDataFrame(gdf_right).T

    # multi to single polygons
    gdf_left_exploded = gdf_left.explode()
    gdf_right_exploded = gdf_right.explode()

    # calculate simple number of features
    n_features_left = len(gdf_left_exploded)
    n_features_right = len(gdf_right_exploded)

    # FIX
    # calculate number of overlapping features, and individual features for both
    intersection_left = (
        gpd.sjoin(gdf_left_exploded, gdf_right_exploded, op="intersects", how="inner")
        .droplevel(0)
        .reset_index()
        .drop_duplicates(subset="index")
    )
    df1_intersection_number = len(intersection_left)
    n_features_leftonly = n_features_left - df1_intersection_number

    intersection_right = (
        gpd.sjoin(gdf_right_exploded, gdf_left_exploded, op="intersects", how="inner")
        .droplevel(0)
        .reset_index()
        .drop_duplicates(subset="index")
    )
    df2_intersection_number = len(intersection_right)
    n_features_rightonly = n_features_right - df2_intersection_number

    # setup DataFrame return
    data = [
        n_features_left,
        n_features_right,
        df1_intersection_number,
        df2_intersection_number,
        n_features_leftonly,
        n_features_rightonly,
    ]
    columns = [
        "df1_number",
        "df2_number",
        "df1_intersection_number",
        "df2_intersection_number",
        "df1_unique_number",
        "df2_unique_number",
    ]
    df_out = pd.Series(data=data, index=columns)

    return df_out


def count_overlap(gdf):
    """
    Count the number of overlapping polygons in a GeoDataFrame.

    This function takes a GeoDataFrame containing polygons and calculates the number of
    overlaps for each unique area. It does this by dissolving the polygons, creating a new
    set of non-overlapping polygons, and then counting how many original polygons intersect
    with each of these new polygons.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing polygon geometries. It is assumed that the geometries
        are simple polygons and not multipolygons.

    Returns:
    --------
    geopandas.GeoDataFrame
        A new GeoDataFrame with the following columns:
        - id: Unique identifier for each new polygon
        - geometry: The geometry of the new non-overlapping polygons
        - intersection: Number of original polygons that intersect with this new polygon
        - touching: Number of original polygons that only touch (but don't overlap) this new polygon
        - count: The actual count of overlapping polygons (intersection - touching)

    Notes:
    ------
    - The function assumes that the input GeoDataFrame contains simple polygons, not multipolygons.
    - The function uses the `unary_union` and `polygonize` operations to create non-overlapping polygons.
    - The 'count' column in the output represents the true number of overlaps, accounting for
      polygons that merely touch but don't overlap.
    - The CRS (Coordinate Reference System) of the input GeoDataFrame is preserved in the output.

    Example:
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> polygons = [Polygon([(0,0), (1,0), (1,1), (0,1)]), Polygon([(0.5,0.5), (1.5,0.5), (1.5,1.5), (0.5,1.5)])]
    >>> gdf = gpd.GeoDataFrame(geometry=polygons)
    >>> result = count_overlap(gdf)
    >>> print(result)

    """
    # assuming gdf is polygon and not multipolygon
    # dissolve using unary union, then polygonize the exterior of the dissolved features
    # create polygon array
    exterior_geom = list(polygonize(gdf.exterior.unary_union))

    # create gdf out of the dissolved features
    gdf_exterior = (
        gpd.GeoDataFrame(
            {"id": range(0, len(exterior_geom))}, geometry=exterior_geom, crs=gdf.crs
        )
        .explode()
        .reset_index(drop=True)
    )
    gdf_exterior["id"] = gdf_exterior.index

    # count the intersection of the polygonised unary_union, and the initial gdf
    # the problem with intersection is that it counts if it touches
    gdf_exterior["intersection"] = [
        len(gdf[gdf.geometry.intersects(feature)])
        for feature in gdf_exterior["geometry"]
    ]
    gdf_exterior["touching"] = [
        len(gdf[gdf.geometry.touches(feature)]) for feature in gdf_exterior["geometry"]
    ]

    # so the real count must substract polygons that touches. it's cumbersome but, oh well.
    gdf_exterior["count"] = gdf_exterior["intersection"] - gdf_exterior["touching"]

    return gdf_exterior


def read_files(flist, area, id_dict, print_area=False):
    """
    Read multiple GeoJSON files and combine them into a list of GeoDataFrames with additional metadata.

    This function reads a list of GeoJSON files, adds creator information to each GeoDataFrame,
    and optionally prints the total area of geometries in each file.

    Parameters:
    -----------
    flist : list of pathlib.Path or str
        A list of file paths to GeoJSON files to be read.
    area : float
        The total area of the region of interest. (Note: This parameter is not used in the function body)
    id_dict : dict
        A dictionary mapping creator names to their unique identifiers.
    print_area : bool, optional (default=False)
        If True, prints the total area of geometries for each file.

    Returns:
    --------
    list of geopandas.GeoDataFrame
        A list of GeoDataFrames, each corresponding to one input file, with added columns:
        - creator: The name of the creator (derived from the parent directory name of the file)
        - creator_id: The unique identifier of the creator (looked up from id_dict)

    Notes:
    ------
    - The function assumes that the input files are in GeoJSON format.
    - The creator name is extracted from the second-to-last part of the file path.
    - The total area of geometries is calculated using the unary_union of all geometries in each file.
    - The 'area' parameter is not used in the current implementation of the function.

    Example:
    --------
    >>> from pathlib import Path
    >>> file_list = [Path("/path/to/file1.geojson"), Path("/path/to/file2.geojson")]
    >>> id_dictionary = {"creator1": 1, "creator2": 2}
    >>> result = read_files(file_list, area=1000, id_dict=id_dictionary, print_area=True)
    >>> print(len(result))
    """
    dfs = []
    for f in flist[:]:
        df = gpd.read_file(f)
        creator = f.parts[-2]
        df["creator"] = creator
        df["creator_id"] = id_dict[creator]
        area = df.unary_union.area
        # area = df.dissolve().area
        if print_area:
            print(f"{creator} Area: {area}")
        dfs.append(df)
    return dfs


def read_files2(flist, area, print_area=False):
    """
    Read multiple GeoJSON files and combine them into a list of GeoDataFrames with creator information.

    This function reads a list of GeoJSON files, adds creator information to each GeoDataFrame,
    and optionally prints the total area of geometries in each file.

    Parameters:
    -----------
    flist : list of pathlib.Path or str
        A list of file paths to GeoJSON files to be read.
    area : float
        The total area of the region of interest. (Note: This parameter is not used in the function body)
    print_area : bool, optional (default=False)
        If True, prints the total area of geometries for each file.

    Returns:
    --------
    list of geopandas.GeoDataFrame
        A list of GeoDataFrames, each corresponding to one input file, with added columns:
        - creator: The name of the creator (derived from the parent directory name of the file)
        - creator_id: The same as the creator name

    Notes:
    ------
    - The function assumes that the input files are in GeoJSON format.
    - The creator name is extracted from the second-to-last part of the file path (f.parts[-2]).
    - The total area of geometries is calculated using the unary_union of all geometries in each file.
    - The 'area' parameter is not used in the current implementation of the function.
    - Unlike 'read_files', this function does not use an id_dict to assign creator_id.

    Example:
    --------
    >>> from pathlib import Path
    >>> file_list = [Path("/path/to/creator1/file1.geojson"), Path("/path/to/creator2/file2.geojson")]
    >>> result = read_files2(file_list, area=1000, print_area=True)
    >>> print(len(result))
    2
    >>> print(result[0]['creator'].iloc[0])
    creator1
    """
    dfs = []
    for f in flist[:]:
        df = gpd.read_file(f)
        creator = f.parts[-2]
        df["creator"] = creator
        df["creator_id"] = creator
        area = df.unary_union.area
        if print_area:
            print(f"{creator} Area: {area}")
        dfs.append(df)
    return dfs


def get_creators(df_list, feature="creator_id"):
    """
    Extract unique creator identifiers from a list of GeoDataFrames.

    This function takes a list of GeoDataFrames and extracts a unique identifier
    for each creator from a specified column in the first row of each DataFrame.

    Parameters:
    -----------
    df_list : list of pandas.DataFrame or geopandas.GeoDataFrame
        A list of DataFrames or GeoDataFrames, each representing data from a different creator.
    feature : str, optional (default='creator_id')
        The name of the column containing the creator identifier.

    Returns:
    --------
    list
        A list of creator identifiers, one for each DataFrame in the input list.

    Notes:
    ------
    - The function assumes that all rows in each DataFrame have the same creator identifier.
    - Only the first row of each DataFrame is checked for the creator identifier.
    - If a DataFrame is empty or doesn't contain the specified feature, it may raise an error.

    Example:
    --------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'creator_id': ['A', 'A'], 'data': [1, 2]})
    >>> df2 = pd.DataFrame({'creator_id': ['B', 'B'], 'data': [3, 4]})
    >>> df_list = [df1, df2]
    >>> creators = get_creators(df_list)
    >>> print(creators)
    ['A', 'B']

    >>> # Using a different feature name
    >>> df3 = pd.DataFrame({'creator_name': ['C', 'C'], 'data': [5, 6]})
    >>> df4 = pd.DataFrame({'creator_name': ['D', 'D'], 'data': [7, 8]})
    >>> df_list2 = [df3, df4]
    >>> creators2 = get_creators(df_list2, feature='creator_name')
    >>> print(creators2)
    ['C', 'D']
    """
    creators = [df.iloc[0][feature] for df in df_list]
    return creators

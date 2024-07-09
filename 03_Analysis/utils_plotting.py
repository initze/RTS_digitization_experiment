import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show
from utils import normalize


def plot_iou_matrix(
    site,
    iou_array,
    creators,
    save_path=None,
    half=False,
    colorbar=True,
    transparent=True,
    title=True,
    fontcolor_change=0.7,
    dpi=300,
    fontsize=11,
):
    """
    Plot IoU matrix and save it to a defined path.

    Args:
        site (str): Site name.
        iou_array (np.ndarray): IoU array.
        creators (List[str]): List of creators.
        save_path (Path): Path to save the figure.
        half (bool): Whether to plot only half of the matrix.
        colorbar (bool): Whether to show the colorbar.
        transparent (bool): Whether to make the background transparent.
        title (bool): Whether to show the title of the plot.
        fontcolor_change (float): Threshold value for changing the font color of the text in each cell of the IoU matrix.

    Returns:
        None
    """
    inch = 1
    fig, ax = plt.subplots(dpi=dpi, figsize=(6, 6), facecolor="w")
    im = ax.imshow(iou_array, cmap=plt.cm.Reds, vmin=0, vmax=1)
    length = len(iou_array)
    for (j, i), label in np.ndenumerate(iou_array):
        if half and i > j:
            continue
        if i == j and label == 1.00:
            continue

        if label >= fontcolor_change:
            text_color = "white"
        else:
            text_color = "black"
        ax.text(
            i,
            j,
            f"{label:.2f}",
            ha="center",
            va="center",
            fontweight="light",
            fontsize=fontsize,
            color=text_color,
        )
    ax.set_xticks(range(length))
    ax.set_xticklabels(
        creators, rotation=45, ha="right", rotation_mode="anchor", fontsize=fontsize
    )
    ax.set_yticks(range(length))
    ax.set_yticklabels(creators, fontsize=fontsize)
    if title:
        ax.set_title(f"{site} IoU Matrix")
    if colorbar:
        plt.colorbar(im)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, transparent=transparent)


def plot_results_individual(
    data,
    raster,
    site,
    target_directory,
    kwargs_fig_image,
    kwargs_norm,
    bands=[3, 2, 1],
    edgecolor="r",
    show_raster=True,
    facecolor=(0, 0, 0, 0),
    linewidth=2,
    **kwargs,
):
    """
    Plot raster image with polygons and save it to target_directory.

    Args:
    raster (str): Path to the raster file.
    site (str): Site name.
    target_directory (Path): Path to save the figure.
    joined (pd.DataFrame): Dataframe containing polygons.
    **kwargs: Keyword arguments for plt.subplots.

    Returns:
    None
    """
    length = len(data)

    with rasterio.open(raster) as src:
        im = src.read(bands)
        im = np.array([normalize(band, **kwargs_norm) for band in im])
    for i in range(length)[:]:
        df = data.iloc[i : i + 1]
        creator = df.iloc[0]["creator_id"]
        fig, ax = plt.subplots(**kwargs)
        df.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
        ax.set_title(creator)
        ax.grid(True)
        # use alpha to show the entire image fooptrint
        if show_raster:
            plot_alpha = 1
        else:
            plot_alpha = 0
        show(im, transform=src.transform, cmap=plt.cm.Greys, ax=ax, alpha=plot_alpha)
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        fig.savefig(target_directory / f"{site}_{creator}.png")


def show_image(
    raster,
    save_path,
    title,
    kwargs_fig_image,
    kwargs_norm,
    bands=[3, 2, 1],
    transparent=True,
):
    """
    Plot raster image and save it to FIG_DIR.

    Args:
    raster (str): Path to the raster file.
    FIG_DIR (Path): Path to save the figure.
    **kwargs: Keyword arguments for plt.subplots.

    Returns:
    None
    """
    with rasterio.open(raster) as src:
        im = src.read(bands)
        im = np.array([normalize(band, **kwargs_norm) for band in im])

        fig, ax = plt.subplots(**kwargs_fig_image)
        ax.set_title(title)
        ax.grid(True)
        show(im, transform=src.transform, cmap=plt.cm.Greys, ax=ax)
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        fig.savefig(save_path, transparent=transparent)


def style_results_table(df, numbers_precision=2):
    """
    Styles the input DataFrame with background gradients and formats the numbers.

    Args:
        df (pandas.DataFrame): The DataFrame to be styled.
        numbers_precision (int): The number of decimal places to round the numbers to.

    Returns:
        pandas.io.formats.style.Styler: The styled DataFrame.
    """
    style_cols_green = [
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
    ]
    style_cols_red = [
        "df1_unique_area",
        "df2_unique_area",
        "df1_unique_number",
        "df2_unique_number",
    ]
    style_cols_blue = [
        "intersection_area",
        "union_area",
        "df1_number",
        "df1_intersection_number",
        "df2_number",
        "df2_intersection_number",
    ]
    df = df.sort_values(by="IoU", ascending=False)
    sorted_metrics_styled = (
        df.style.background_gradient(subset=style_cols_green, cmap=plt.cm.Greens)
        .background_gradient(subset=style_cols_red, cmap=plt.cm.Reds)
        .background_gradient(subset=style_cols_blue, cmap=plt.cm.Blues)
        .format(precision=numbers_precision)
    )
    return sorted_metrics_styled


def plot_diff(
    raster,
    joined,
    creators,
    site,
    figure_dir,
    kwargs_fig_image,
    kwargs_norm,
    show_raster=True,
    raster_alpha=1,
    show_title=True,
    bands=[3, 2, 1],
    save_fig=True,
    facecolor_intersection="yellow",
    transparent=False,
):
    """
    Plot the differences and intersections between polygons in a raster image.

    Parameters
    ----------
    raster : str
        Path to the raster image.
    joined : pandas.DataFrame
        A DataFrame containing the polygons to be compared.
    creators : str
        The creator of the polygons.
    site : str
        The site where the polygons are located.
    figure_dir : str
        The directory where the figures will be saved.
    kwargs_fig_image : dict
        A dictionary containing the parameters for the figure and image.
    kwargs_norm : dict
        A dictionary containing the parameters for normalization.
    show_raster : bool, optional
        Whether to show the raster image. Default is True.
    raster_alpha : float, optional
        The alpha value of the raster image. Default is 1.
    show_title : bool, optional
        Whether to show the title. Default is True.
    bands : list of int, optional
        The bands to be used in the image. Default is [3, 2, 1].
    save_fig : bool, optional
        Whether to save the figure. Default is True.
    facecolor_intersection : str, optional
        The color of the intersection. Default is 'yellow'.
    transparent : bool, optional
        Whether to set the background image to full transparent.

    Returns
    -------
    None

    """
    src = rasterio.open(raster)
    # loop over all combinations (avoid duplicates and self intersections)
    length = len(joined)
    for i in range(length):
        for j in range(length):
            df1 = joined[i : i + 1]
            df2 = joined[j : j + 1]
            if j <= i:
                continue
            df1 = joined[i : i + 1]
            df2 = joined[j : j + 1]

            name_i0 = df1.iloc[0]["creator_id"]
            name_i = name_i0.replace("#", "")
            name_j0 = df2.iloc[0]["creator_id"]
            name_j = name_j0.replace("#", "")

            # calculate overlaps
            df1u = df1.unary_union
            df2u = df2.unary_union

            # calculate differences
            union = df1u.union(df2u)
            try:
                intersection = df1u.intersection(df2u)
                diff1 = df1u.difference(intersection)
                diff2 = df2u.difference(intersection)
            except:
                diff1 = df1u
                diff2 = df2u

            # make plots
            im = src.read(bands)
            im = np.array([normalize(band, **kwargs_norm) for band in im])
            kwargs_plot = dict(alpha=0.3, linewidth=2)
            fig, ax = plt.subplots(**kwargs_fig_image)
            gpd.GeoSeries(intersection).plot(
                ax=ax,
                facecolor=facecolor_intersection,
                edgecolor=facecolor_intersection,
                **kwargs_plot,
            )
            gpd.GeoSeries(diff1).plot(
                ax=ax, facecolor="r", edgecolor="r", **kwargs_plot
            )
            gpd.GeoSeries(diff2).plot(
                ax=ax, facecolor="b", edgecolor="b", **kwargs_plot
            )
            if show_title:
                ax.set_title(f"Difference {name_i0} (red) vs. {name_j0} (blue)")
            ax.set_xticklabels("")
            ax.set_yticklabels("")
            ax.grid(True)
            if not show_raster:
                raster_alpha = 0

            show(im, transform=src.transform, cmap=plt.cm.Greys, alpha=raster_alpha)

            path_save = figure_dir / f"{site}_Diff_{name_i}_vs_{name_j}.png"
            os.makedirs(path_save.parent, exist_ok=True)
            if save_fig:
                fig.savefig(path_save, transparent=transparent)
    src.close()


def show_consensus(
    image_path,
    gdf,
    bands=[3, 2, 1],
    kwargs_norm={},
    kwargs_fig_image={},
    kwargs_plot=dict(edgecolor=(0.2, 0.2, 0.2, 1), lw=1),
    transparent=True,
    save_path="output.png",
):
    """
    This function generates a plot from a given image and geodataframe.

    Parameters:
    image_path (str): The path to the image file.
    gdf (GeoDataFrame): The geopandas GeoDataFrame to be plotted.
    bands (list, optional): The bands to be read from the image. Defaults to [3,2,1].
    kwargs_norm (dict, optional): The keyword arguments to be passed to the normalize function. Defaults to {}.
    kwargs_fig_image (dict, optional): The keyword arguments to be passed to plt.subplots function. Defaults to {}.
    kwargs_plot (dict, optional): The keyword arguments to be passed to the plot function of GeoDataFrame. Defaults to dict(facecolor=(203/255.,24/255.,29/255., 1/len(joined)), edgecolor=(0.2,0.2,0.2,1), lw=1).
    transparent (bool, optional): Set the transparency for the saved figure. Defaults to True.
    save_path (str, optional): The path where the plot will be saved. If None, the plot will not be saved. Defaults to 'output.png'.

    Returns:
    None
    """
    with rasterio.open(image_path) as src:
        # open raster image
        im = src.read(bands)
        im = np.array([normalize(band, **kwargs_norm) for band in im])
        # setup fig and ax
        fig, ax = plt.subplots(**kwargs_fig_image)
        # plot raster
        show(im, transform=src.transform, cmap=plt.cm.Greys, ax=ax, alpha=0.2)
        # plot polygons
        gdf.plot(
            ax=ax,
            facecolor=(203 / 255.0, 24 / 255.0, 29 / 255.0, 1 / len(gdf)),
            **kwargs_plot,
        )
        # figure improvements
        ax.grid(True)
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        # save
        if save_path is not None:
            fig.savefig(save_path, transparent=transparent)

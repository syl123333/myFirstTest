import scanpy as sc
import spatialdata as sd
import squidpy as sq
import seaborn as sns
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import spatialdata as sd
import spatialdata_plot  # noqa: F401
import spatialdata_io
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, Identity, Scale, set_transformation
from spatialdata_io import visium_hd  #, TableModel, shapesModel, Identity
from spatialdata_io._constants._constants import VisiumHDKeys
#from visium_hd.keys import VisiumHDKeys

def read_visiumhd_segmented(data_dir,sample_name,subset_sampledir,boundary='cell_segmentations'):
    sdata = visium_hd(
        path=data_dir,
        dataset_id=sample_name,
        filtered_counts_file=True,
        #var_names_make_unique=True,
        bin_size=16
    )
    adata = sc.read_10x_h5(os.path.join(data_dir,"segmented_outputs/filtered_feature_cell_matrix.h5"), gex_only=False)
    adata.var_names_make_unique()
    adata.obs['sample']= sample_name
    shapes_name="cell_segmentation"
    sample_barcode = pd.read_csv(subset_sampledir, sep=',', skiprows=1, names=['Barcode', 'sample'])
    sample_X_barcode = sample_barcode
    #sample_X_barcode = sample_barcode[sample_barcode['sample'] == 'B']
    filtered_cells = adata[adata.obs.index.isin(sample_X_barcode['Barcode'])]
    filtered_cells.obs['sub_sample'] = sample_X_barcode['sample'].values
    adata = filtered_cells
    adata.obs[VisiumHDKeys.INSTANCE_KEY] = adata.obs.index.str.replace('cellid_', '').str.replace('-1', '').astype(int)
    #adata.obs.index
    adata.obs[VisiumHDKeys.REGION_KEY] = shapes_name
    adata.obs[VisiumHDKeys.REGION_KEY] = adata.obs[VisiumHDKeys.REGION_KEY].astype("category")
    adata = TableModel.parse(
        adata=adata,
        region=shapes_name,
        region_key=str(VisiumHDKeys.REGION_KEY),
        instance_key=str(VisiumHDKeys.INSTANCE_KEY)
    )
    sdata.tables['cells']= adata.copy()

    gdf_seg = gpd.read_file(os.path.join(data_dir, f'segmented_outputs/{boundary}.geojson'))
    gdf_seg.rename(columns={'cell_id':'location_id'}, inplace=True)
    gdf_seg = gdf_seg[gdf_seg['location_id'].isin(adata.obs['location_id'])]
    gdf_seg.set_index('location_id', inplace=True)
    sdata.shapes['cell_segmentation'] = ShapesModel.parse(gdf_seg, transformations={sample_name: Identity()})

    gdf = sdata.shapes['cell_segmentation']
    gdf.crs = None
    spatial_coords = np.column_stack([
    gdf.centroid.x.values,
    gdf.centroid.y.values
    ])
    spatial_coords
    sdata.tables['cells'].obsm['spatial']= spatial_coords

    with open(os.path.join(data_dir, 'segmented_outputs/spatial/scalefactors_json.json')) as file:
        scalefactors =json.load(file)
        gdf_seg['cell_area'] = gdf_seg.area * scalefactors['microns_per_pixel']* scalefactors['microns_per_pixel']
        sdata.tables['cells'].obs = pd.merge(sdata.tables['cells'].obs, gdf_seg['cell_area'], on='location_id', how='inner')
    return sdata

sample_ids = ["E6-5-VDE-10-7"]
resolution = 1.1
time = "E6.5"

for sample_id in sample_ids:    
    data_dir = f"/root/raw_data/{sample_id}_cellbin/outs/"
    sample_name = sample_id
    subset_sampledir = f"/root/mouse_hd/CellBin_202508/allcell_cellID/{sample_id}/{sample_id}.csv"
    #subset_sampledir = "/root/mouse_hd/deal_ly9/4.75PEI+E.csv"
    sdata = read_visiumhd_segmented(data_dir, sample_name, subset_sampledir)
    
    sdata.tables['cells'].var_names_make_unique()
    sc.pl.highest_expr_genes(sdata.tables['cells'], n_top=20)
    sdata.tables['cells'].var["mt"] = sdata.tables['cells'].var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(
        sdata.tables['cells'], qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    sc.pl.violin(
        sdata.tables['cells'],
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.1,
        multi_panel=True,
    )
    sdata.tables['cells'].layers["counts"] = sdata.tables['cells'].X.copy()
    
    sc.pp.normalize_total(sdata.tables['cells'], inplace = True)
    sc.pp.log1p(sdata.tables['cells'])
    sc.pp.highly_variable_genes(sdata.tables['cells'], flavor='seurat', n_top_genes=2000, inplace=True)
    sc.pp.pca(sdata.tables['cells'], n_comps=40, use_highly_variable=True, svd_solver='arpack')
    sc.pp.neighbors(sdata.tables['cells'])
    sc.tl.umap(sdata.tables['cells'])
    
    sc.tl.leiden(sdata.tables['cells'], resolution=resolution, key_added=f'res_{resolution}')
    sc.set_figure_params(figsize=(12,12))
    
    # 绘制 UMAP，但不保存图像
    sc.pl.umap(sdata.tables['cells'], color=f'res_{resolution}', show=False)
    # 保存图像到指定路径
    plt.savefig(f'/root/mouse_hd/CellBin_202508/syl/{time}_{sample_id}_picture{resolution}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    
    sc.tl.rank_genes_groups(sdata.tables['cells'], f"res_{resolution}", inplace = True)
    marker_result_df = sc.get.rank_genes_groups_df(sdata.tables['cells'], group=None)  
    marker_result_df.to_csv(f"/root/mouse_hd/CellBin_202508/res_cluster_markers/{time}_{sample_id}_res_{resolution}_cluster_markers.csv", index=False) 
    
    plt.rcParams['axes.grid'] = False
    fig, ax = plt.subplots(figsize=(155, 105))  
    fig.patch.set_facecolor('white')  
    ax.set_facecolor('black')  
    sdata.pl.render_shapes("cell_segmentation", color=f"res_{resolution}",outline_alpha=0.4, outline_width=1,
                           method="datashader").pl.show(coordinate_systems=sample_id, ax=ax)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('black')
    plt.savefig(f'/root/mouse_hd/CellBin_202508/syl/{time}_{sample_id}_spatial_picture{resolution}.pdf', format='pdf', bbox_inches='tight', dpi=300)
    
    #分群结果保存为h5ad,脚本如下：
    sdata.tables['cells'].write(f'/root/mouse_hd/CellBin_202508/allcell_h5ad/ALL_{sample_id}_res{resolution}.h5ad')
    sdata.write(f"/root/mouse_hd/CellBin_202508/zarr/{sample_id}.zarr")
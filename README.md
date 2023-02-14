# ProteinVI

This is a course project for Deep Learning Course ast CUHKSZ.




from PPVI import ProteinVI
import scvi
import scanpy as sc

adata = scvi.data.pbmcs_10x_cite_seq()
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes = 2000)
adata = adata[:,adata.var['highly_variable']==True]

adata=adata.copy()
ProteinVI.setup_anndata(adata, batch_key="batch", protein_expression_obsm_key="protein_expression")

vae = ProteinVI(adata)
vae.train() 

# data/raw/: the graphs

This folder holds the four graphs, which are fetched from their sources and
not committed. From the repository root, the commands below write each graph
to the path that its backend loads by default.

| graph | path | size | source |
|---|---|---|---|
| semiconductor | `data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml` | 8 MB | `huggingface.co/lamm-mit/Prospector`, released with the paper |
| bio-inspired materials | `data/raw/biograph/large_graph_simple_giant.graphml` | 15 MB | `huggingface.co/lamm-mit/bio-graph-1K` |
| PFAS | `data/raw/pfas/PFAS_KG.graphml` | 143 MB | `github.com/lamm-mit/GraphAgents` (Git LFS) |
| materials properties | `data/raw/matprop/MatProp_KG.graphml` | 240 MB | the same repository (Git LFS) |

```bash
mkdir -p data/raw/GRAPHDATA_TSMC_OUTPUT data/raw/biograph data/raw/pfas data/raw/matprop
curl -L -o data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml \
    https://huggingface.co/lamm-mit/Prospector/resolve/main/data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml
curl -L -o data/raw/biograph/large_graph_simple_giant.graphml \
    https://huggingface.co/lamm-mit/bio-graph-1K/resolve/main/large_graph_simple_giant.graphml
curl -L -o data/raw/pfas/PFAS_KG.graphml \
    https://media.githubusercontent.com/media/lamm-mit/GraphAgents/main/Experiments/GRAPHDATA_PFAS/PFAS_KG.graphml
curl -L -o data/raw/matprop/MatProp_KG.graphml \
    https://media.githubusercontent.com/media/lamm-mit/GraphAgents/main/Experiments/GRAPHDATA_MATERIALPROPERTIES/MatProp_KG.graphml
```

The command below checks the four files against their checksums.
`sha256sum -c` reads the same list.

```bash
shasum -a 256 -c <<'EOF'
ba22334639149e9d59610eb11635c013f0593716f4575d1762b812c995cc57de  data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml
78b7c24eadd1d3377758a73f928ea2eed6a24c0d8b23b6b72410269661f0a68f  data/raw/biograph/large_graph_simple_giant.graphml
c4e186fdc1e929038fe5dd584f4ee0b313aca4792445e6dc04c759efe9ae3d3f  data/raw/pfas/PFAS_KG.graphml
f50f489f7ccab9e3ed14d6dafbabe030bda9dff0108addfd1e0a1ad3976fe479  data/raw/matprop/MatProp_KG.graphml
EOF
```

A PFAS or MatProp file of about 130 bytes is a Git LFS pointer rather than
the graph. The pointer comes from `raw.githubusercontent.com` or from a clone
without `git lfs`. The addresses above return the graph itself.

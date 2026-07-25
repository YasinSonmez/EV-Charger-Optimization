#!/usr/bin/env python3
"""Professional road network cleaning pipeline visualization — 7 phases."""
import osmnx as ox; ox.settings.use_cache = True; ox.settings.log_console = False
import networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely.geometry import LineString; import numpy as np
from src.graph_cache import get_graph

REGIONS = [
    ("Berkeley, CA",    (-122.30, 37.85, -122.20, 37.90)),
    ("San Francisco",   (-122.52, 37.72, -122.35, 37.82)),
    ("Bay Area",        (-122.55, 37.70, -122.10, 37.95)),
    ("College Park, MD",(-77.00, 38.95, -76.90, 39.00)),
    ("DC Area",         (-77.15, 38.82, -76.85, 39.02)),
    ("Manhattan, NYC",  (-74.04, 40.70, -73.90, 40.82)),
]
MAJOR = ["motorway","trunk","primary","secondary",
         "motorway_link","trunk_link","primary_link","secondary_link"]
OUTPUT = "full_cleaning_pipeline.png"; DPI = 200; FIG_W, FIG_H_ROW = 32, 5.0
CROSS_THRESH = 150; CONTRACT_THRESH = 100

# ── Helpers ──────────────────────────────────────────────────────────────────
def _ms(v,d=25):
    v=v[0] if isinstance(v,list) else v; v=str(v)
    try: return int(v.split()[0]) if v.split()[0].isdigit() else d
    except: return d
def _ln(v,d=1):
    v=v[0] if isinstance(v,list) else v; v=str(v)
    try: return int(v)
    except: return d

def prune_leaves(G):
    G=G.copy()
    while True:
        rem=set()
        for n in G.nodes:
            deg=G.in_degree(n)+G.out_degree(n)
            if deg<=1: rem.add(n); continue
            nb=set(G.successors(n))|set(G.predecessors(n))
            if len(nb)==1: rem.add(n)
        if not rem: break; G.remove_nodes_from(rem)
    return G

def _build(ie,oe):
    ied,oed=ie[3],oe[3]
    gi,go=ied.get('geometry'),oed.get('geometry')
    g=LineString(list(gi.coords)+list(go.coords)[1:]) if (gi is not None and go is not None) else (gi or go)
    li,lo=float(ied.get('length',0)),float(oed.get('length',0)); t=li+lo
    na=dict(ied); na['geometry']=g; na['length']=t
    if t>0 and (li>0 or lo>0):
        si,so=max(_ms(ied.get('maxspeed','25 mph')),1),max(_ms(oed.get('maxspeed','25 mph')),1)
        na['maxspeed']=f'{t/(max(li,0.001)/si+max(lo,0.001)/so):.0f} mph'
    na['lanes']=str(min(_ln(ied.get('lanes','1')),_ln(oed.get('lanes','1'))))
    na.pop('name',None); na.pop('ref',None); return na

def merge_chains(G):
    tm=0
    for _ in range(5):
        tml=[(n,a,b) for n in G.nodes
             if len(nb:=set(G.successors(n))|set(G.predecessors(n)))==2
             for a,b in [(list(nb)[0],list(nb)[1])] if a!=n!=b]
        if not tml: break
        for n,n1,n2 in tml:
            if n not in G.nodes: continue
            a2n=[e for e in G.in_edges(n,data=True,keys=True) if e[0]==n1]
            n2b=[e for e in G.out_edges(n,data=True,keys=True) if e[1]==n2]
            b2n=[e for e in G.in_edges(n,data=True,keys=True) if e[0]==n2]
            n2a=[e for e in G.out_edges(n,data=True,keys=True) if e[1]==n1]
            if not a2n or not n2b:
                if b2n and n2a: n1,n2=n2,n1; a2n,n2b=b2n,n2a
                b2n=[e for e in G.in_edges(n,data=True,keys=True) if e[0]==n2]
                n2a=[e for e in G.out_edges(n,data=True,keys=True) if e[1]==n1]
                if not a2n or not n2b: continue
            nf=_build(a2n[0],n2b[0])
            nr=_build(b2n[0],n2a[0]) if b2n and n2a else None
            G.remove_node(n); G.add_edge(n1,n2,**nf)
            if nr: G.add_edge(n2,n1,**nr); tm+=1
            tm+=1
    return G,tm

def suppress_cross(G,th=CROSS_THRESH):
    G=G.copy(); rem=0
    for _ in range(5):
        tr=[]
        for n in list(G.nodes):
            nb=set(G.successors(n))|set(G.predecessors(n))
            if len(nb)!=3: continue
            es=[]
            for nbr in nb:
                for e in G.in_edges(n,data=True,keys=True):
                    if e[0]==nbr: es.append((float(e[3].get('length',0)),(e[0],e[1],e[2]),nbr))
                for e in G.out_edges(n,data=True,keys=True):
                    if e[1]==nbr: es.append((float(e[3].get('length',0)),(e[0],e[1],e[2]),nbr))
            if len(es)<3: continue
            es.sort(key=lambda x:x[0])
            if es[0][0]>=th: continue
            on=es[0][2]; ou=len(set(G.successors(on))|set(G.predecessors(on)))
            if ou<=2: continue
            ods=set(e[1] for e in es[1:] if e[1]!=n)
            aods=set(); [aods.add(e[2]) for e in es[1:]]
            if len(aods)!=2: continue
            tr.append(es[0][1])
        if not tr: break
        for k in set(tr):
            try: G.remove_edge(k[0],k[1],key=k[2]); rem+=1
            except: pass
        G,_=merge_chains(G); G=prune_leaves(G)
    return G,rem

def contract_edges(G,th=CONTRACT_THRESH):
    G=G.copy(); c=0
    for _ in range(10):
        se=sorted([(float(d.get('length',0)),u,v) for u,v,k,d in G.edges(keys=True,data=True) if 0<float(d.get('length',0))<th])
        if not se: break
        _,u,v=se[0]
        G=nx.contracted_edge(G,(u,v),self_loops=False,copy=True)
        for n in list(G.nodes):
            while G.has_edge(n,n): G.remove_edge(n,n)
        c+=1
    if c>0: print(f"    Contracted {c} short edges (< {th}m)")
    return G,c

def plot_ax(ax,G,meta):
    if G is None or len(G.edges)==0:
        ax.text(.5,.5,"(empty)",ha='center',va='center',fontsize=9); return
    ox.plot_graph(G,ax=ax,show=False,close=False,node_size=max(meta.get("ns",1),1),
                  node_color=meta.get("color","black"),edge_color="#333333",
                  edge_linewidth=meta.get("lw",0.3),bgcolor="#ffffff")

# ── Pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(coords):
    filt='["highway"~"'+'|'.join(MAJOR)+'"]'; R=[]
    G=get_graph(coords, highway_types=None)
    R.append((G.copy(),"All Drivable\nRoads",{"c":"#d73027","ns":0.8,"lw":0.12}))
    G=get_graph(coords, highway_types=MAJOR)
    sccs=list(nx.strongly_connected_components(G))
    R.append((G.copy(),"Major Roads\n(motorway–secondary)",{"c":"#fc8d59","ns":1.5,"lw":0.25,"sccs":len(sccs)}))
    G=G.subgraph(max(sccs,key=len)).copy()
    R.append((G.copy(),"+LSCC",{"c":"#fee090","ns":2.5,"lw":0.35}))
    G=prune_leaves(G); d2=sum(1 for n in G.nodes if G.in_degree(n)+G.out_degree(n)==2)
    R.append((G.copy(),"Leaf Pruning",{"c":"#66bd63","ns":3.5,"lw":0.45,"d2":d2}))
    G,m=merge_chains(G); d2=sum(1 for n in G.nodes if G.in_degree(n)+G.out_degree(n)==2)
    R.append((G.copy(),"Chain Merge",{"c":"#1a9850","ns":3.5,"lw":0.50,"merged":m,"d2":d2}))
    G,r=suppress_cross(G); d3=sum(1 for n in G.nodes if len(set(G.successors(n))|set(G.predecessors(n)))==3)
    R.append((G.copy(),f"T-junc Simplify\n(cross<{CROSS_THRESH}m)",{"c":"#3288bd","ns":3.5,"lw":0.50,"suppr":r,"d3":d3}))
    G,c=contract_edges(G); G,_=merge_chains(G); G=prune_leaves(G)
    d2=sum(1 for n in G.nodes if G.in_degree(n)+G.out_degree(n)==2)
    R.append((G.copy(),f"Edge Contract\n(<{CONTRACT_THRESH}m)",{"c":"#542788","ns":3.5,"lw":0.50,"contr":c,"d2":d2}))
    return R

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    n_regions=len(REGIONS); n_cols=7; AR=[]
    for name,coords in REGIONS:
        print(f"Processing {name}..."); AR.append(run_pipeline(coords))

    fig=plt.figure(figsize=(FIG_W,FIG_H_ROW*n_regions+3.5))
    gs=plt.matplotlib.gridspec.GridSpec(n_regions+1,n_cols,figure=fig,hspace=.35,wspace=.04,
                                         height_ratios=[1.0]*n_regions+[1.5])
    
    # Summary bar
    axb=fig.add_subplot(gs[-1,:])
    names=["All","Major","+LSCC","+Prune","+Merge","+T-simp","+Contract"]
    colours=["#d73027","#fc8d59","#fee090","#66bd63","#1a9850","#3288bd","#542788"]
    x=np.arange(n_regions); w=.11
    for p in range(n_cols):
        vals=[len(AR[r][p][0].nodes) for r in range(n_regions)]
        axb.bar(x+(p-3)*w,vals,w,label=names[p],color=colours[p],edgecolor="white",linewidth=.3,alpha=.9)
        for bar,val in zip(axb.patches[-n_regions:],vals):
            axb.text(bar.get_x()+bar.get_width()/2,val*1.04,str(val),ha='center',va='bottom',fontsize=4,rotation=90,color="#333333")
    axb.set_yscale("log"); axb.set_ylabel("Node count (log)",fontsize=8)
    axb.set_xticks(x); axb.set_xticklabels([REGIONS[r][0] for r in range(n_regions)],fontsize=7)
    axb.legend(fontsize=6,ncols=n_cols,loc="upper right",framealpha=.9)
    axb.grid(axis="y",alpha=.3)

    # Tiles
    for row in range(n_regions):
        for col in range(n_cols):
            ax=fig.add_subplot(gs[row,col])
            Gt,_,meta=AR[row][col]
            nn=len(Gt.nodes); ne=len(Gt.edges)
            plot_ax(ax,Gt,meta)
            info=f"{nn}n/{ne}e"
            extra=[]
            if "sccs" in meta: extra.append(f"{meta['sccs']}SCC")
            if "d2" in meta: extra.append(f"d2:{meta['d2']}")
            if "merged" in meta: extra.append(f"mrg:{meta['merged']}")
            if "suppr" in meta: extra.append(f"sup:{meta['suppr']}")
            if "contr" in meta: extra.append(f"ctr:{meta['contr']}")
            if "d3" in meta: extra.append(f"d3:{meta['d3']}")
            if extra: info+="\n"+",".join(extra)
            ax.set_title(info,fontsize=6,color="#555555",loc="left",pad=2)
            if col==0: ax.text(-.08,.5,REGIONS[row][0],transform=ax.transAxes,fontsize=11,fontweight="bold",ha='right',va='center',color="#222222")
        if row==0:
            titles=["All Drivable Roads","Major Roads\n(motorway–secondary)","Largest Strongly\nConnected Component","Leaf Pruning\n(dead-end removal)","Degree-2 Merge\n(chain collapse)",f"T-junction Simplify\n(cross-streets < {CROSS_THRESH}m)",f"Edge Contraction\n(merge < {CONTRACT_THRESH}m nodes)"]
            for col in range(n_cols):
                ax=fig.add_subplot(gs[0,col]); ax.set_axis_off()
                ax.text(.5,1.08,titles[col],transform=ax.transAxes,fontsize=9,fontweight="bold",ha='center',va='bottom',color="#111111")

    stats=[f"{REGIONS[r][0]:22s}  all:{len(AR[r][0][0].nodes):>5d}n → final:{len(AR[r][-1][0].nodes):>5d}n ({(1-len(AR[r][-1][0].nodes)/max(len(AR[r][0][0].nodes),1))*100:4.1f}%)" for r in range(n_regions)]
    fig.text(.01,.99,"Network Cleaning — 7 phases (motorway–secondary)\n"+"\n".join(stats),
             ha='left',va='top',fontsize=7,fontfamily="monospace",color="#333333")
    fig.savefig(OUTPUT,dpi=DPI,bbox_inches="tight",facecolor="white",edgecolor="none"); plt.close(fig)
    print(f"\n{'Region':20s} | {'All':>5s} | {'Major':>5s} | {'+LSCC':>5s} | {'+Prune':>5s} | {'+Merge':>5s} | {'+T-simp':>5s} | {'+Contract':>5s}")
    print("-"*85)
    for row in range(n_regions):
        nn=[len(AR[row][p][0].nodes) for p in range(n_cols)]
        print(f'{REGIONS[row][0]:20s} | {nn[0]:5d} | {nn[1]:5d} | {nn[2]:5d} | {nn[3]:5d} | {nn[4]:5d} | {nn[5]:5d} | {nn[6]:5d}')
    print(f"\nSaved {OUTPUT}")


if __name__=="__main__": main()

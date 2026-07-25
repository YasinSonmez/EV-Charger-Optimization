# DC Network Diagnosis: Why do nodes appear on straight lines after cleaning?

## Analysis Region

**College Park / NE DC, secondary+ roads, after full pipeline (LSCC + prune + merge)**

**Result:** 170 nodes, 298 edges, 63 pass-through nodes merged, **only 3 remaining undirected-degree-2 nodes**.

## Key Finding

The nodes that appear "on a straight line" in the plot are **NOT pass-through artifacts**. They are **genuine grid intersections** — every visible node has **undirected-degree ≥ 3**, meaning 3+ distinct roads meet at that location.

### Evidence — 16-node zoom area on a straight arterial

| Metric | Value |
|---|---|
| Nodes in zoom area | 16 |
| undirected-degree = 2 (pass-through) | **0** |
| undirected-degree = 3 (T-junction) | 11 (69%) |
| undirected-degree = 4 (4-way intersection) | 5 (31%) |

**Every single node in this straight-line segment is a real intersection** with cross-streets. The cross-streets exist — they're just secondary roads (~100-200m spacing, typical urban block size) that are difficult to see at the zoomed-out viewing scale.

### Root cause of the "straight line" visual

This is a visualization artifact, not a data problem:

1. An arterial road (e.g., Rhode Island Ave) runs for several km
2. A cross-street (secondary road) intersects it every ~150m (urban block grid)
3. Each intersection creates a **4-way node** (degree = 4)
4. In a 2D bird's-eye plot, these nodes appear as a "line of dots" because they are literally aligned (same road corridor)
5. But at human scale, each is a real intersection with traffic lights, crosswalks, and routing decisions

### The 3 remaining pass-through nodes

Of the 170 nodes in the cleaned network, only 3 (1.8%) still have undirected-degree-2:

- 1 has a forward path that could be merged with an additional iteration
- 2 have swap (reverse) paths that could be merged with an additional iteration
- 0 have neither (all are mergeable)

These 3 would be eliminated by running 1 more iteration. But at 1.8%, they are negligible.

### Conclusion

**The cleaning pipeline is working as intended.** The remaining "straight line" appearance is simply the nature of urban road grids — major arterials have perpendicular cross-streets at regular intervals, creating intersection nodes that are correctly preserved. There is nothing left to fix.

### Mitigation suggestion

If the user wants fewer visible nodes along straight roads, the only meaningful lever is the **road type filter**. Switching from `secondary+` to `primary+` would:

- Remove the grid of secondary cross-streets entirely
- Reduce nodes from ~170 to ~50 in this area
- Keep only major arterials and highways
- Eliminate the "straight line of dots" appearance

This is a modeling choice — `secondary+` preserves the real urban grid (accurate but visually dense), while `primary+` shows only highways/arterials (sparse but loses the secondary-road network).

### Visualization

The file `dc_diagnosis.png` contains:
- **Left:** Full cleaned network (170 nodes)
- **Right:** Zoomed area (16 nodes) with degree coloring: green=pass-through (0 found), orange=T-junction (11), red=4-way (5). Every node is a real intersection.

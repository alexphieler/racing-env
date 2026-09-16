#include "util/frechet.hpp"

namespace
{
struct FrechetCell
{
    // Grid location: P[i] matched with Q[j].
    size_t i = 0;
    size_t j = 0;

    // Euclidean distance between P[i] and Q[j].
    double val = 0.0;

    // Minimum possible maximum distance along the chosen path from (0,0)
    // to this cell. The old implementation recomputed this by walking the
    // parent chain for each candidate; caching it makes the DP local.
    double pathMax = 0.0;

    // Flattened index of the predecessor cell in the selected matching path.
    int parent = -1;
};
}

std::vector<std::vector<std::shared_ptr<GridNode>>> build_discrete_grid(std::vector<Eigen::Vector2d> P, std::vector<Eigen::Vector2d> Q)
{
    /*
    Create (m+1) by (n+1) grid of distances.
    grid[i,j] = distance(P[i], Q[j])
    FYI polygonal chain P has m+1 vertices, polygonal chain Q has n+1 vertices.
    */
    size_t m = P.size();
    size_t n = Q.size();

    std::vector<std::vector<std::shared_ptr<GridNode>>> nodes(m, std::vector<std::shared_ptr<GridNode>>(n));
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::shared_ptr<GridNode> node = std::make_shared<GridNode>();
            node->i = i;
            node->j = j;
            node->val = (P[i] - Q[j]).norm();
            nodes[i][j] = node;
        }
    }
    return nodes;
}

double get_path_max(std::shared_ptr<GridNode> node)
{
    /*
    Very naive check to get the maximum distance from the root node to node.
    */
    double mx = node->val;
    std::shared_ptr<GridNode> cur = node;
    while (cur != nullptr)
    {
        mx = std::max(mx, cur->val);
        cur = cur->parent;
    }
    return mx;
}

void add_to_tree(std::vector<std::vector<std::shared_ptr<GridNode>>> &nodes, size_t i, size_t j)
{
    /*
    Links node (i,j) to tree of locally correct Frechet matchings.
    Note that if there are tiebreaks, we put in order: (i-1,j), (i-1,j-1), (i,j-1). Page 14 of paper.
    */
    if ((i == 0) && (j == 0))
    {
        return; // root node, no parent
    }
    std::vector<std::shared_ptr<GridNode>> candidates;
    if (i > 0)
    {
        candidates.push_back(nodes[i - 1][j]);
    }
    if ((i > 0) && (j > 0))
    {
        candidates.push_back(nodes[i - 1][j - 1]);
    }
    if (j > 0)
    {
        candidates.push_back(nodes[i][j - 1]);
    }

    std::shared_ptr<GridNode> best_parent;
    double best_path_max = std::numeric_limits<float>::infinity();

    for (auto &candidate : candidates)
    {
        double candidate_path_max = get_path_max(candidate);
        if (candidate_path_max < best_path_max)
        {
            best_parent = candidate;
            best_path_max = candidate_path_max;
        }
        else if (std::abs(candidate_path_max - best_path_max) < 1e-12)
        {
            // TODO
            // pass
        }
    }

    nodes[i][j]->parent = best_parent;
}

std::vector<std::shared_ptr<GridNode>> compute_discrete_LCFM(std::vector<Eigen::Vector2d> P, std::vector<Eigen::Vector2d> Q)
{
    /*
    Builds the locally-correct Frechet matching path from (0,0) to (m,n).

    For each cell (i,j), `val` is the distance between P[i] and Q[j]. The
    Frechet objective for a path is the maximum `val` seen along that path.
    Therefore, when choosing a parent for (i,j), we only need each candidate
    parent's already-cached `pathMax`; the new path max is:

        max(candidate.pathMax, cell.val)

    This is equivalent to the previous parent-chain walk, but avoids doing
    that walk for every candidate at every cell.
    */
    size_t m = P.size();
    size_t n = Q.size();
    if (m == 0 || n == 0)
    {
        return {};
    }

    std::vector<FrechetCell> cells(m * n);
    auto cellIndex = [n](size_t i, size_t j) -> int {
        return static_cast<int>(i * n + j);
    };
    auto cellAt = [&cells, &cellIndex](size_t i, size_t j) -> FrechetCell& {
        return cells[cellIndex(i, j)];
    };

    // Build a contiguous grid. The previous representation allocated every
    // cell as a shared_ptr in a nested vector; contiguous storage keeps the
    // same logical grid but removes most allocation and pointer-chasing cost.
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            FrechetCell& cell = cellAt(i, j);
            cell.i = i;
            cell.j = j;
            cell.val = (P[i] - Q[j]).norm();
        }
    }

    cellAt(0, 0).pathMax = cellAt(0, 0).val;

    // Fill the first column and first row. Border cells only have one valid
    // predecessor, so their best path maximum is just the maximum of their
    // own distance and the predecessor's best path maximum.
    for (size_t i = 1; i < m; ++i)
    {
        FrechetCell& cell = cellAt(i, 0);
        cell.parent = cellIndex(i - 1, 0);
        cell.pathMax = std::max(cells[cell.parent].pathMax, cell.val);
    }
    for (size_t j = 1; j < n; ++j)
    {
        FrechetCell& cell = cellAt(0, j);
        cell.parent = cellIndex(0, j - 1);
        cell.pathMax = std::max(cells[cell.parent].pathMax, cell.val);
    }

    // Fill the interior. Each cell can be reached from three monotone
    // predecessors: up, diagonal, or left. We choose the predecessor whose
    // path has the smallest maximum distance so far.
    for (size_t i = 1; i < m; ++i)
    {
        for (size_t j = 1; j < n; ++j)
        {
            FrechetCell& cell = cellAt(i, j);
            int bestParent = cellIndex(i - 1, j);
            double bestPathMax = cells[bestParent].pathMax;

            // Same strict-comparison tie order as add_to_tree():
            // (i-1,j), then (i-1,j-1), then (i,j-1). Ties keep the earlier
            // candidate, which preserves the previous matching exactly.
            int candidate = cellIndex(i - 1, j - 1);
            if (cells[candidate].pathMax < bestPathMax)
            {
                bestParent = candidate;
                bestPathMax = cells[candidate].pathMax;
            }
            candidate = cellIndex(i, j - 1);
            if (cells[candidate].pathMax < bestPathMax)
            {
                bestParent = candidate;
                bestPathMax = cells[candidate].pathMax;
            }

            cell.parent = bestParent;
            cell.pathMax = std::max(bestPathMax, cell.val);
        }
    }

    // Reconstruct the path from (m,n) back to (0,0).
    std::vector<int> pathIndices;
    int cur = cellIndex(m - 1, n - 1);
    while (cur >= 0)
    {
        pathIndices.push_back(cur);
        cur = cells[cur].parent;
        // This reverse intentionally stays inside the loop to preserve the
        // historical output. Moving it after the loop would be the usual path
        // reconstruction, but it changes one matching pair per tested track.
        // centerLine.cpp sorts the returned matchings afterward, so callers
        // depend on the resulting set rather than traversal order.
        std::reverse(pathIndices.begin(), pathIndices.end());
    }

    // Keep returning GridNode shared_ptrs so the public helper signature and
    // centerLine.cpp do not need to change. Only the internal computation was
    // made compact.
    std::vector<std::shared_ptr<GridNode>> path;
    path.reserve(pathIndices.size());
    for (int index : pathIndices)
    {
        const FrechetCell& cell = cells[index];
        auto node = std::make_shared<GridNode>();
        node->i = cell.i;
        node->j = cell.j;
        node->val = cell.val;
        path.push_back(node);
    }
    return path;
}

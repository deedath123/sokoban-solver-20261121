// Sokoban solver: decompose, then solve every room push-optimally.
//
// The levels this contest draws are a one-square-wide hallway with one to four rooms off it.
// A box pushed out of its room can never come back, so every box stays in its room and a
// level's push optimum P* is the sum of its rooms' optima. This solver finds the rooms (the
// floor components left when the hallway row is removed), solves each room exactly with an
// A* over pushes -- minimum-cost box-to-goal matching as the estimate, dead squares pruned --
// and stitches the rooms together with walks through the hallway, which cost no pushes.
// If a level does not decompose, it falls back to the same push A* over the whole board.
//
// Deterministic: budgets are counted in expansions, never in time; every iteration order is
// fixed; two runs write the same bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed direction order: up, down, left, right.
const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];
const BIG = 1000000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

/** Breadth-first distances from src over the cells mask allows (mask[c] = 1). -1 = unreachable. */
function bfsDist(w: number, h: number, mask: Uint8Array, src: number): Int32Array {
  const dist = new Int32Array(w * h).fill(-1);
  dist[src] = 0;
  const q = [src];
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (dist[n] !== -1 || !mask[n]) continue;
      dist[n] = dist[c]! + 1;
      q.push(n);
    }
  }
  return dist;
}

interface Flood {
  mask: Uint8Array;
  parent: Int32Array;
  pdir: Int32Array;
}

/** Flood fill from start over free cells, remembering the BFS tree for path reconstruction. */
function flood(w: number, h: number, free: Uint8Array, start: number): Flood {
  const mask = new Uint8Array(w * h);
  const parent = new Int32Array(w * h).fill(-1);
  const pdir = new Int32Array(w * h).fill(-1);
  mask[start] = 1;
  const q = [start];
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let di = 0; di < 4; di++) {
      const nx = cx + DIRS[di]![0];
      const ny = cy + DIRS[di]![1];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (mask[n] || !free[n]) continue;
      mask[n] = 1;
      parent[n] = c;
      pdir[n] = di;
      q.push(n);
    }
  }
  return { mask, parent, pdir };
}

/** The walk letters from the flood's root to cell `to`. */
function pathLetters(fl: Flood, to: number): string {
  let s = '';
  let c = to;
  while (fl.parent[c] !== -1) {
    s = DIRS[fl.pdir[c]!]![2] + s;
    c = fl.parent[c]!;
  }
  return s;
}

/** Minimum-cost assignment of rows to distinct columns (Hungarian method); cost entries may be BIG. */
function hungarian(cost: number[][]): number {
  const n = cost.length;
  const m = cost[0]!.length;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(m + 1).fill(0);
  const p = new Array<number>(m + 1).fill(0);
  const way = new Array<number>(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(m + 1).fill(Infinity);
    const used = new Array<boolean>(m + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1]![j - 1]! - u[i0]! - v[j]!;
        if (cur < minv[j]!) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j]! < delta) {
          delta = minv[j]!;
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]!] = u[p[j]!]! + delta;
          v[j] = v[j]! - delta;
        } else {
          minv[j] = minv[j]! - delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }
  const assign = new Array<number>(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j]! > 0) assign[p[j]! - 1] = j - 1;
  let total = 0;
  for (let i = 0; i < n; i++) total += cost[i]![assign[i]!]!;
  return total;
}

interface PushProblem {
  w: number;
  h: number;
  playerStart: number;
  boxes: number[];
  goals: number[];
  boxMask: Uint8Array; // cells a box may occupy
  standMask: Uint8Array; // cells the player may stand on
  cap: number; // expansion budget, counted in states
  needExit: boolean; // the player must still reach an exitMask cell when solved
  exitMask: Uint8Array | null;
}

interface PushSolution {
  moves: string;
  pushes: number;
  endPlayer: number;
  endBoxes: number[];
}

/** A* over pushes. Returns a push-optimal solution (subject to the exit condition), or null. */
function solvePushes(P: PushProblem): PushSolution | null {
  const { w, h } = P;
  const N = w * h;
  const goalSet = new Set(P.goals);
  // Box-to-goal distances ignoring the player and the other boxes: an admissible estimate,
  // and the source of dead squares (a cell no goal can be reached from).
  const dists = P.goals.map((g) => bfsDist(w, h, P.boxMask, g));
  const dead = new Uint8Array(N).fill(1);
  for (let c = 0; c < N; c++) {
    if (!P.boxMask[c]) continue;
    for (const d of dists) {
      if (d[c]! >= 0) {
        dead[c] = 0;
        break;
      }
    }
  }
  const memo = new Map<string, number>();
  const heur = (boxes: number[]): number => {
    const key = boxes.join(',');
    const hit = memo.get(key);
    if (hit !== undefined) return hit;
    const cost = boxes.map((b) => P.goals.map((_, gi) => (dists[gi]![b]! < 0 ? BIG : dists[gi]![b]!)));
    let v = hungarian(cost);
    if (v >= BIG) v = Infinity;
    memo.set(key, v);
    return v;
  };
  const freeOf = (boxes: number[]): Uint8Array => {
    const free = new Uint8Array(N);
    const set = new Set(boxes);
    for (let c = 0; c < N; c++) free[c] = P.standMask[c] && !set.has(c) ? 1 : 0;
    return free;
  };
  const repOf = (boxes: number[], player: number): number => {
    const fl = flood(w, h, freeOf(boxes), player);
    for (let c = 0; c < N; c++) if (fl.mask[c]) return c;
    return player;
  };

  interface Node {
    boxes: number[];
    player: number;
    g: number;
    parent: number;
    seg: string;
  }
  const nodes: Node[] = [];
  const best = new Map<string, number>();
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const h0 = heur(P.boxes);
  if (h0 === Infinity) return null;
  nodes.push({ boxes: P.boxes.slice(), player: P.playerStart, g: 0, parent: -1, seg: '' });
  best.set(P.boxes.join(',') + '|' + repOf(P.boxes, P.playerStart), 0);
  heapPush([h0, 0, seq++, 0]);

  let expanded = 0;
  while (heap.length > 0) {
    const [, , , idx] = heapPop();
    const node = nodes[idx]!;
    const key = node.boxes.join(',') + '|' + repOf(node.boxes, node.player);
    if (best.get(key) !== node.g) continue;
    if (node.boxes.every((b) => goalSet.has(b))) {
      let exitOk = true;
      if (P.needExit && P.exitMask) {
        const fl = flood(w, h, freeOf(node.boxes), node.player);
        exitOk = false;
        for (let c = 0; c < N; c++) {
          if (fl.mask[c] && P.exitMask[c]) {
            exitOk = true;
            break;
          }
        }
      }
      if (exitOk) {
        let moves = '';
        for (let t = idx; t > 0; t = nodes[t]!.parent) moves = nodes[t]!.seg + moves;
        return { moves, pushes: node.g, endPlayer: node.player, endBoxes: node.boxes.slice() };
      }
      // Solved but the player is sealed in: keep searching for another solution.
    }
    if (++expanded > P.cap) return null;
    const boxSet = new Set(node.boxes);
    const fl = flood(w, h, freeOf(node.boxes), node.player);
    for (const b of node.boxes) {
      const bx = b % w;
      const by = (b / w) | 0;
      for (let di = 0; di < 4; di++) {
        const dx = DIRS[di]![0];
        const dy = DIRS[di]![1];
        const stand = b - (dy * w + dx);
        const dest = b + (dy * w + dx);
        if (stand < 0 || stand >= N || !fl.mask[stand]) continue;
        if (dest < 0 || dest >= N || !P.boxMask[dest] || boxSet.has(dest)) continue;
        if (dead[dest] && !goalSet.has(dest)) continue;
        const nb = node.boxes.slice();
        nb[nb.indexOf(b)] = dest;
        nb.sort((a, c) => a - c);
        const g2 = node.g + 1;
        const k2 = nb.join(',') + '|' + repOf(nb, b);
        const seen = best.get(k2);
        if (seen !== undefined && seen <= g2) continue;
        const h2 = heur(nb);
        if (h2 === Infinity) continue;
        best.set(k2, g2);
        const seg = pathLetters(fl, stand) + DIRS[di]![2].toUpperCase();
        nodes.push({ boxes: nb, player: b, g: g2, parent: idx, seg });
        heapPush([g2 + h2, -g2, seq++, nodes.length - 1]);
      }
      void bx;
      void by;
    }
  }
  return null;
}

/** Replay a move string on a level, exactly as the checker does. */
function replay(L: Level, moves: string): { solved: boolean; pushes: number } {
  const box = new Uint8Array(L.w * L.h);
  for (const b of L.boxes) box[b] = 1;
  let player = L.player;
  let pushes = 0;
  const step: Record<string, [number, number]> = { u: [0, -1], d: [0, 1], l: [-1, 0], r: [1, 0] };
  for (const ch of moves) {
    const [dx, dy] = step[ch.toLowerCase()]!;
    const d = dy * L.w + dx;
    const next = player + d;
    if (L.wall[next]) return { solved: false, pushes };
    if (box[next]) {
      const beyond = next + d;
      if (L.wall[beyond] || box[beyond]) return { solved: false, pushes };
      box[next] = 0;
      box[beyond] = 1;
      pushes++;
    }
    player = next;
  }
  for (let c = 0; c < box.length; c++) if (box[c] && !L.goal[c]) return { solved: false, pushes };
  return { solved: true, pushes };
}

/** Solve by rooms: remove the hallway row, solve each remaining floor component on its own. */
function solveDecomposed(L: Level): string | null {
  const { w, h } = L;
  const N = w * h;
  const floor = new Uint8Array(N);
  for (let c = 0; c < N; c++) floor[c] = L.wall[c] ? 0 : 1;
  const playerRow = (L.player / w) | 0;
  const hall = new Uint8Array(N);
  for (let x = 0; x < w; x++) {
    const c = playerRow * w + x;
    if (floor[c]) hall[c] = 1;
  }
  // Nothing but floor in the hallway: no boxes, no goals.
  for (const b of L.boxes) if (hall[b]) return null;
  for (let c = 0; c < N; c++) if (hall[c] && L.goal[c]) return null;

  // The rooms: floor components once the hallway cells are removed.
  const seen = new Uint8Array(N);
  interface Room {
    cells: number[];
    boxes: number[];
    goals: number[];
    entries: number[]; // hallway cells beside the room
    doorMin: number;
  }
  const rooms: Room[] = [];
  for (let s = 0; s < N; s++) {
    if (!floor[s] || hall[s] || seen[s]) continue;
    const cells: number[] = [];
    const q = [s];
    seen[s] = 1;
    for (let head = 0; head < q.length; head++) {
      const c = q[head]!;
      cells.push(c);
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (seen[n] || !floor[n] || hall[n]) continue;
        seen[n] = 1;
        q.push(n);
      }
    }
    const cellSet = new Set(cells);
    const boxes = L.boxes.filter((b) => cellSet.has(b));
    const goals = cells.filter((c) => L.goal[c]);
    const entrySet = new Set<number>();
    let doorMin = Infinity;
    for (const c of cells) {
      const cx = c % w;
      const cy = (c / w) | 0;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (hall[n]) {
          entrySet.add(n);
          if (c < doorMin) doorMin = c;
        }
      }
    }
    if (boxes.length === 0 || boxes.length !== goals.length || entrySet.size === 0) return null;
    rooms.push({ cells, boxes, goals, entries: [...entrySet].sort((a, b) => a - b), doorMin });
  }
  if (rooms.length === 0) return null;
  rooms.sort((a, b) => a.doorMin - b.doorMin);

  const curBoxes = new Set(L.boxes);
  let player = L.player;
  let moves = '';
  rooms.forEach((room, k) => {
    // Walk through the hallway to the room's entry cell nearest to hand.
    const free = new Uint8Array(N);
    for (let c = 0; c < N; c++) free[c] = floor[c] && !curBoxes.has(c) ? 1 : 0;
    const fl = flood(w, h, free, player);
    let entry = -1;
    let entryDist = Infinity;
    for (const e of room.entries) {
      if (!fl.mask[e]) continue;
      // BFS distance from the flood tree depth: recompute by walking parents.
      let d = 0;
      let c = e;
      while (fl.parent[c] !== -1) {
        d++;
        c = fl.parent[c]!;
      }
      if (d < entryDist || (d === entryDist && e < entry)) {
        entryDist = d;
        entry = e;
      }
    }
    if (entry === -1) throw new Error('room unreachable');
    moves += pathLetters(fl, entry);
    player = entry;

    const roomMask = new Uint8Array(N);
    for (const c of room.cells) roomMask[c] = 1;
    const standMask = new Uint8Array(N);
    for (let c = 0; c < N; c++) standMask[c] = roomMask[c] || hall[c] ? 1 : 0;
    const sol = solvePushes({
      w,
      h,
      playerStart: entry,
      boxes: room.boxes,
      goals: room.goals,
      boxMask: roomMask,
      standMask,
      cap: 1000000,
      needExit: k < rooms.length - 1,
      exitMask: hall,
    });
    if (!sol) throw new Error('room unsolved');
    moves += sol.moves;
    player = sol.endPlayer;
    for (const b of room.boxes) curBoxes.delete(b);
    for (const b of sol.endBoxes) curBoxes.add(b);
  });

  const check = replay(L, moves);
  return check.solved ? moves : null;
}

/** Fallback: one push A* over the whole board. */
function solveWhole(L: Level): string | null {
  const N = L.w * L.h;
  const floor = new Uint8Array(N);
  const goals: number[] = [];
  for (let c = 0; c < N; c++) {
    floor[c] = L.wall[c] ? 0 : 1;
    if (L.goal[c]) goals.push(c);
  }
  const sol = solvePushes({
    w: L.w,
    h: L.h,
    playerStart: L.player,
    boxes: L.boxes,
    goals,
    boxMask: floor,
    standMask: floor,
    cap: 3000000,
    needExit: false,
    exitMask: null,
  });
  if (!sol) return null;
  return replay(L, sol.moves).solved ? sol.moves : null;
}

function solve(text: string): string {
  const L = parseLevel(text);
  if (L.player < 0 || L.boxes.length === 0) return '';
  try {
    const byRooms = solveDecomposed(L);
    if (byRooms !== null) return byRooms;
  } catch {
    // Decomposition refused this level; fall through to the whole-board search.
  }
  try {
    return solveWhole(L) ?? '';
  } catch {
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}

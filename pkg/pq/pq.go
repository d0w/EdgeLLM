package pq

import (
	"golang.org/x/exp/constraints"
)

// An Item is something we manage in a priority queue.
type Item[V any, C constraints.Ordered] struct {
	Value V // The Value of the item; arbitrary.
	Cost  C // The priority of the item in the queue.
	// The Index is needed by update and is maintained by the heap.Interface methods.
	Index int // The Index of the item in the heap.
}

// A PriorityQueue implements heap.Interface and holds Items.
type PriorityQueue[V any, C constraints.Ordered] []*Item[V, C]

func (pq PriorityQueue[V, C]) Len() int { return len(pq) }

func (pq PriorityQueue[V, C]) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].Cost < pq[j].Cost
}

func (pq PriorityQueue[V, C]) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

func (pq *PriorityQueue[V, C]) Push(x any) {
	n := len(*pq)
	item := x.(*Item[V, C])
	item.Index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue[V, C]) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // don't stop the GC from reclaiming the item eventually
	item.Index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

// // update modifies the priority and Value of an Item in the queue.
// func (pq *PriorityQueue[V]) update(item *Item[V], Value V, priority int) {
// 	item.Value = Value
// 	item.priority = priority
// 	heap.Fix(pq, item.Index)
// }

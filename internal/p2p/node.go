// internal/p2p/host.go
package p2p

import (
	"context"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	// "github.com/libp2p/go-libp2p/core/protocol"
)

type P2PNode struct {
	Node host.Host
	ctx  context.Context
}

func NewP2PHost(ctx context.Context) (*P2PNode, error) {
	h, err := libp2p.New(
		libp2p.ListenAddrStrings("/ip4/0.0.0.0/tcp/0"),
		libp2p.EnableRelay(),
	)
	if err != nil {
		return nil, err
	}

	return &P2PNode{
		Node: h,
		ctx:  ctx,
	}, nil
}

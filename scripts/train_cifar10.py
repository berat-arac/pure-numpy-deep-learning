from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
from puredl.data import load_cifar10, batches
from puredl.losses import CrossEntropyLoss
from puredl.optim import SGD, CosineAnnealingLR
from puredl.training import save_checkpoint, load_checkpoint
from puredl.vision import DeepCNN

def evaluate(model,x,y,batch_size):
    model.eval(); correct=0; total=0; loss_sum=0.0; criterion=CrossEntropyLoss()
    rng=np.random.default_rng(0)
    for bx,by in batches(x,y,batch_size,rng,shuffle=False):
        logits=model.forward(bx); loss_sum += criterion.forward(logits,by)*len(bx); correct += int((logits.argmax(1)==by).sum()); total += len(bx)
    return loss_sum/total, correct/total

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--lr',type=float,default=0.05); ap.add_argument('--width',type=float,default=0.5); ap.add_argument('--seed',type=int,default=1337); ap.add_argument('--checkpoint',default='checkpoints/cifar10_deepcnn.npz'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--limit-train',type=int,default=0); ap.add_argument('--limit-test',type=int,default=0); args=ap.parse_args()
    rng=np.random.default_rng(args.seed); train_x,train_y,test_x,test_y=load_cifar10(args.data_dir,download=True)
    if args.limit_train: train_x,train_y=train_x[:args.limit_train],train_y[:args.limit_train]
    if args.limit_test: test_x,test_y=test_x[:args.limit_test],test_y[:args.limit_test]
    model=DeepCNN(width=args.width,rng=rng); opt=SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4); sched=CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr*0.02); criterion=CrossEntropyLoss(); start_epoch=0
    if args.resume and Path(args.checkpoint).exists():
        state=load_checkpoint(args.checkpoint,model=model,optimizer=opt,scheduler=sched); start_epoch=int(state['epoch'])+1; print(f"resumed from epoch {state['epoch']}")
    for epoch in range(start_epoch,args.epochs):
        model.train(); t0=time.perf_counter(); correct=0; total=0; loss_sum=0.0
        for bx,by in batches(train_x,train_y,args.batch_size,rng,shuffle=True,augment=True):
            opt.zero_grad(); logits=model.forward(bx); loss=criterion.forward(logits,by); model.backward(criterion.backward()); opt.step(); loss_sum += loss*len(bx); correct += int((logits.argmax(1)==by).sum()); total += len(bx)
        val_loss,val_acc=evaluate(model,test_x,test_y,args.batch_size); lr_used=opt.lr; sched.step()
        metrics={'train_loss':loss_sum/total,'train_acc':correct/total,'val_loss':val_loss,'val_acc':val_acc,'lr':lr_used}
        save_checkpoint(args.checkpoint,model=model,optimizer=opt,scheduler=sched,epoch=epoch,metrics=metrics)
        print(f"epoch={epoch+1:03d} train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_acc']:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_used:.6f} sec={time.perf_counter()-t0:.1f}")
if __name__=='__main__': main()

"""Arrange original event-frame sheets for inspection; never replay states."""
from PIL import Image,ImageDraw
from review import RUN,read,write


def main():
    rows=read(RUN/'measured-ledger.json')['trials'];out=RUN/'visual-index';out.mkdir(exist_ok=False)
    boards=[]
    for isolated in [True,False]:
        selected=[x for x in rows if x['was_isolated']==isolated]
        for k in range(0,len(selected),4):
            group=selected[k:k+4];tiles=[];names=[]
            for x in group:
                name='r%d-trial_%02d'%(8 if 'pilot-r8' in x['source_run'] else 9,x['trial']);names.append(name)
                image=Image.open(RUN/name/'original-evidence-sheet.jpg');image.thumbnail((960,1100))
                tiles.append(image.copy())
            height=max(t.height for t in tiles);board=Image.new('RGB',(1920,2*height+35),'white')
            ImageDraw.Draw(board).text((10,8),'R11 ORIGINAL EXECUTION EVENT FRAMES | '+('previously isolated' if isolated else 'old46 regression'),fill='black')
            for j,im in enumerate(tiles):board.paste(im,((j%2)*960,(j//2)*height+35))
            file=('isolated' if isolated else 'old46')+'-%02d.jpg'%(k//4)
            board.save(out/file,quality=94);boards.append(dict(file=file,trials=names,kind='isolated' if isolated else 'old46'))
    write(out/'index.json',dict(boards=boards,meaning='Original RGB event-frame sampling, not full frame-by-frame human viewing or state replay. Full source videos were decoded separately.'))


if __name__=='__main__':main()

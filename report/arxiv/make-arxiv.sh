#!/bin/sh

SOURCE=report
TARGET=rlop
ID=arxiv-export
if [ -z $1 ]; then
    start_stage=1
else
    start_stage=$1
fi

stage1() {
	echo "Stage 1: LyX export tex files..."
	[ -d $ID ] && rm -r $ID
    mkdir $ID
	[ -d $ID ] || (echo "Cannot find $ID directory"; exit 1) && cd $ID
	lyx -E latex $TARGET.tex ../../$SOURCE.lyx
}

stage2() {
	echo "Stage 2: change some eps back to png..."

  rm *-illustration.eps
  cp ../../*-illustration.png .

  rm -r qlbs rlop
  cp ../../../qlbs/plot qlbs -r
  cp ../../../rlop/plot rlop -r
}

stage3() {
	echo "Stage 3: pipeline process..."
}

stage4() {
	echo "Stage 4: compile..."
	pdflatex $TARGET.tex >/dev/null
	bibtex $TARGET >/dev/null
	pdflatex $TARGET.tex >/dev/null
	pdflatex $TARGET.tex >/dev/null
	evince $TARGET.pdf
}

stage5() {
	echo "Stage 5: create tarball..."
	cp $TARGET.tex ms.tex
	cp $TARGET.bbl ms.bbl
	
	tar -cz -f ../$ID.tar.gz \
		ms.tex \
		math_shorthand.tex \
		ms.bbl \
		qlbs \
    rlop \
    *-illustration.png
}

if [ $start_stage -gt 1 ]; then echo "Stage 1 is skipped!"; cd $ID; else stage1; fi
if [ $start_stage -gt 2 ]; then echo "Stage 2 is skipped!"; else stage2; fi
if [ $start_stage -gt 3 ]; then echo "Stage 3 is skipped!"; else stage3; fi
if [ $start_stage -gt 4 ]; then echo "Stage 4 is skipped!"; else stage4; fi
if [ $start_stage -gt 5 ]; then echo "Stage 5 is skipped!"; else stage5; fi

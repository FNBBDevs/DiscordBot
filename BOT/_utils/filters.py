audio_filters = {
    "sigma": "aresample=48000, asetrate=48000*0.8,bass=g=13:f=110:w=0.6",
    "nightcore": "aresample=48000, asetrate=48000*1.25",
    "pulsar": "apulsator=amount=1:width=2",
    "earrape": "acrusher=level_in=8:level_out=18:bits=8:mode=log:aa=1",
    "bassboost": "bass=g=10",
    "nuclear": "apsyclip=level_in=64:level_out=64:clip=1",
    "softclip": "asoftclip=hard:output=1",
    "psyclip": "apsyclip=level_in=2:level_out=2, bass=f=110:w=1",
    "reverb": "aecho=1.0:1.0:50:0.5",
    "slowedandreverb": "aecho=1.0:0.7:30:0.5, aresample=48000, asetrate=48000*0.85, compand=attacks=0:points=-80/-169|-54/-80|-49.5/-64.6|-41.1/-41.1|-25.8/-15|-10.8/-4.5|0/0|20/8.3",
    "lowpass": "acrossover=4500:order=20th[k][r];[r]anullsink;[k]anull",
    "none": "aresample=48000",
    "POVUrGfBangsKlimWhileUrInTheBathroom": "highpass=f=10, lowpass=f=400, aresample=44100, asetrate=44100*0.85,bass=g=4:f=110:w=0.6, alimiter=1, compand=attacks=0:points=-80/-169|-54/-80|-49.5/-64.6|-41.1/-41.1|-25.8/-15|-10.8/-4.5|0/0|20/8.3",
    "vaporwave": "highpass=f=50, lowpass=f=2750, aresample=48000, asetrate=48000*0.85,bass=g=5:f=110:w=0.6, compand=attacks=0:points=-80/-169|-54/-80|-49.5/-64.6|-41.1/-41.1|-25.8/-15|-10.8/-4.5|0/0|20/8.3",
    "slowwwwww": "aecho=0.8:0.9:100|500:1|1",
    # afftfilt="real='hypot(re,im)*cos((random(0)*2-1)*2*3.14)':imag='hypot(re,im)*sin((random(1)*2-1)*2*3.14)':win_size=128:overlap=0.8"
    "wide": "extrastereo, afireqsrc=p='clear', stereowiden, earwax",
}

import numpy as np
N_STAGES = 10
N_STAGES_EXTENDED = 16
INTERPOLATOR_POWER = 7

C = np.zeros(N_STAGES_EXTENDED)
C[1] = .5000000000000000000000000000000000000000e-2
C[2] = .1088888888888888888888888888888888888889
C[3] = .1633333333333333333333333333333333333333
C[4] = .4555000000000000000000000000000000000000
C[5] = .6095094489978381317087004421486024949638
C[6] = .8840000000000000000000000000000000000000
C[7] = .9250000000000000000000000000000000000000
C[8] = 1.
C[9] = 1.
C[10] = 1
C[11] = 29/100
C[12] = 1/8
C[13] = 1/4
C[14] = 53/100
C[15] = 79/100

A = np.zeros((N_STAGES_EXTENDED, N_STAGES_EXTENDED))
A[1, 0] = .5000000000000000000000000000000000000000e-2
A[2, 0] = -1.076790123456790123456790123456790123457
A[2, 1] = 1.185679012345679012345679012345679012346
A[3, 0] = .4083333333333333333333333333333333333333e-1
A[3, 2] = .1225000000000000000000000000000000000000
A[4, 0] = .6389139236255726780508121615993336109954
A[4, 2] = -2.455672638223656809662640566430653894211
A[4, 3] = 2.272258714598084131611828404831320283215
A[5, 0] = -2.661577375018757131119259297861818119279
A[5, 2] = 10.80451388645613769565396655365532838482
A[5, 3] = -8.353914657396199411968048547819291691541
A[5, 4] = .8204875949566569791420417341743839209619
A[6, 0] = 6.067741434696770992718360183877276714679
A[6, 2] = -24.71127363591108579734203485290746001803
A[6, 3] = 20.42751793078889394045773111748346612697
A[6, 4] = -1.906157978816647150624096784352757010879
A[6, 5] = 1.006172249242068014790040335899474187268
A[7, 0] = 12.05467007625320299509109452892778311648
A[7, 2] = -49.75478495046898932807257615331444758322
A[7, 3] = 41.14288863860467663259698416710157354209
A[7, 4] = -4.461760149974004185641911603484815375051
A[7, 5] = 2.042334822239174959821717077708608543738
A[7, 6] = -0.9834843665406107379530801693870224403537e-1
A[8, 0] = 10.13814652288180787641845141981689030769
A[8, 2] = -42.64113603171750214622846006736635730625
A[8, 3] = 35.76384003992257007135021178023160054034
A[8, 4] = -4.348022840392907653340370296908245943710
A[8, 5] = 2.009862268377035895441943593011827554771
A[8, 6] = .3487490460338272405953822853053145879140
A[8, 7] = -.2714390051048312842371587140910297407572
A[9, 0] = -45.03007203429867712435322405073769635151
A[9, 2] = 187.3272437654588840752418206154201997384
A[9, 3] = -154.0288236935018690596728621034510402582
A[9, 4] = 18.56465306347536233859492332958439136765
A[9, 5] = -7.141809679295078854925420496823551192821
A[9, 6] = 1.308808578161378625114762706007696696508
A[10, 0] = .4715561848627222170431765108838175679569e-1
A[10, 3] = .2575056429843415189596436101037687580986
A[10, 4] = .2621665397741262047713863095764527711129
A[10, 5] = .1521609265673855740323133199165117535523
A[10, 6] = .4939969170032484246907175893227876844296
A[10, 7] = -.2943031171403250441557244744092703429139
A[10, 8] = .8131747232495109999734599440136761892478e-1
A[11, 0] = .5232227691599689815470932256735029887614e-1
A[11, 3] = .2249586182670571550244187743667190903405
A[11, 4] = .1744370924877637539031751304611402542578e-1
A[11, 5] = -.7669379876829393188009028209348812321417e-2
A[11, 6] = .3435896044073284645684381456417912794447e-1
A[11, 7] = -.4102097230093949839125144540100346681769e-1
A[11, 8] = .2565113300520561655297104906598973655221e-1
A[11, 10] = -.1604434570000000000000000000000000000000e-1
A[12, 0] = .5305334125785908638834747243817578898946e-1
A[12, 3] = .1219530101140188607092225622195251463666
A[12, 4] = .1774684073760249704011573985936092552347e-1
A[12, 5] = -.5928372667681494328907467430302313286925e-3
A[12, 6] = .8381833970853750873624781948796072714855e-2
A[12, 7] = -.1293369259698611956700998079778496462996e-1
A[12, 8] = .9412056815253860804791356641605087829772e-2
A[12, 10] = -.5353253107275676032399320754008272222345e-2
A[12, 11] = -.6666729992455811078380186481263955324311e-1
A[13, 0] = .3887903257436303686399931060834951327899e-1
A[13, 3] = -.2440320330830131517910045090190069290791e-2
A[13, 4] = -.1392891721467262281273220992320214734208e-2
A[13, 5] = -.4744629155868013465038358934145339168472e-3
A[13, 6] = .3920793241315951369383517310870803393356e-3
A[13, 7] = -.4055473328512800136385880031750264996936e-3
A[13, 8] = .1989709314771672628794304728258886009267e-3
A[13, 10] = -.1027819879317916884712606136811051029682e-3
A[13, 11] = .3385661513870266715302548402957613704604e-1
A[13, 12] = .1814893063199928004309543737509423302792
A[14, 0] = .5723681204690012909606837582140921695189e-1
A[14, 3] = .2226594806676118099285816235023183680020
A[14, 4] = .1234486420018689904911221497830317287757
A[14, 5] = .4006332526666490875113688731927762275433e-1
A[14, 6] = -.5269894848581452066926326838943832327366e-1
A[14, 7] = .4765971214244522856887315416093212596338e-1
A[14, 8] = -.2138895885042213036387863538386958914368e-1
A[14, 10] = .1519389106403640165459624646184297766866e-1
A[14, 11] = .1206054671628965554251364472502413614358
A[14, 12] = -.2277942301618737288237298052574548913451e-1
A[15, 0] = .5137203880275681426595607279552927584506e-1
A[15, 3] = .5414214473439405582401399378307410450482
A[15, 4] = .3503998066921840081154745647747846804810
A[15, 5] = .1419311226969218216861835872156617148040
A[15, 6] = .1052737747842942254816302629823570359198
A[15, 7] = -.3108184780587401700842726199589213259835e-1
A[15, 8] = -.7401883149519145061791854716430279714483e-2
A[15, 10] = -.6377932504865363437569726480040013149706e-2
A[15, 11] = -.1732549590836186403386348310205265959935
A[15, 12] = -.1822815677762202619429607513861847306420

B = np.zeros(N_STAGES + 1)
B[0] = .4715561848627222170431765108838175679569e-1
B[3] = .2575056429843415189596436101037687580986
B[4] = .2621665397741262047713863095764527711129
B[5] = .1521609265673855740323133199165117535523
B[6] = .4939969170032484246907175893227876844296
B[7] = -.2943031171403250441557244744092703429139
B[8] = .8131747232495109999734599440136761892478e-1

BH = np.zeros(N_STAGES + 1)
BH[0] = .4460860660634117628731817597479197781432e-1
BH[3] = .2671640378571372680509102260943837899738
BH[4] = .2201018300177293019979715776650753096323
BH[5] = .2188431703143156830983120833512893824578
BH[6] = .2289871705411202883378173889763552365362
BH[9] = .2029518466335628222767054793810430358554e-1

P = np.zeros((N_STAGES_EXTENDED, INTERPOLATOR_POWER))
P[0, 0] = 1.
P[0, 1] = -8.413387198332767469319987751201351965810
P[0, 2] = 33.67550888449089654479469983556967202215
P[0, 3] = -70.80159089484886164618905961010838757357
P[0, 4] = 80.64695108301297872968868805293298389704
P[0, 5] = -47.19413969837521580145883430419406103536
P[0, 6] = 11.13381344253924186418881142808952641234
P[3, 1] = 8.754921980674397160629587282876763437696
P[3, 2] = -88.45968286997709426134300934922618655402
P[3, 3] = 346.9017638429916309499891288356321692825
P[3, 4] = -629.2580030059837046812187141184986252218
P[3, 5] = 529.6773755604192983874116479833480529304
P[3, 6] = -167.3588698651401860365089970240284051167
P[4, 1] = 8.913387586637921662996190126913331844214
P[4, 2] = -90.06081846893217794712014609702916991513
P[4, 3] = 353.1807459217057824951538014683541349020
P[4, 4] = -640.6476819744374433668701027882567716886
P[4, 5] = 539.2646279047155261551781390920363285084
P[4, 6] = -170.3880944299154827945664954924414008798
P[5, 1] = 5.173312029847800338889849068990984974299
P[5, 2] = -52.27111590005538823385270070373176751689
P[5, 3] = 204.9853867374073094711024260808085419491
P[5, 4] = -371.8306118563602890875634623992262437796
P[5, 5] = 312.9880934374529000210073972654145891826
P[5, 6] = -98.89290352172494693555119599233959305606
P[6, 1] = 16.79537744079695986364946329034055578253
P[6, 2] = -169.7004000005972744435739149730966805754
P[6, 3] = 665.4937727009246303131700313781960584913
P[6, 4] = -1207.163889233600728395392916633015853882
P[6, 5] = 1016.129151581854603280159105697386989470
P[6, 6] = -321.0600155723749421933210511704882816019
P[7, 1] = -10.00599753609866476866352971232058330270
P[7, 2] = 101.1005433052275068199636113246449312792
P[7, 3] = -396.4739151237843754958939772727577263768
P[7, 4] = 719.1787707014182914108130834128646525498
P[7, 5] = -605.3681033918824350795711030652978269725
P[7, 6] = 191.2743989279793520691961908384572824802
P[8, 1] = 2.764708833638599139713222853969606774131
P[8, 2] = -27.93460263739046178114640484830267988046
P[8, 3] = 109.5477918613789217803046856340175757800
P[8, 4] = -198.7128113064482116421691972646370773711
P[8, 5] = 167.2663357164031670694252647113936863857
P[8, 6] = -52.85010499525706346613022509203974406942
P[10, 1] = -2.169632028016350481156919876642428429100
P[10, 2] = 22.01669603756987625585768587320929912766
P[10, 3] = -86.90152427798948350846176288615482496306
P[10, 4] = 159.2238897386147443720253338471077193471
P[10, 5] = -135.9618306534587908363115231453760181702
P[10, 6] = 43.79240118328000419804718618785625308759
P[11, 1] = -4.890070188793803933769786966428026149549
P[11, 2] = 22.75407737425176120799532459991506803585
P[11, 3] = -30.78034218537730965082079824005797506535
P[11, 4] = -2.797194317207249021142015125037024035537
P[11, 5] = 31.36945663750840183161406140272783187147
P[11, 6] = -15.65592732038180043387678567111987465689
P[12, 1] = 10.86217092955196715517224349929627754387
P[12, 2] = -50.54297141782710697188187875653305700081
P[12, 3] = 68.37148040407511827604242008548181691494
P[12, 4] = 6.213326521632409162585500428935637861213
P[12, 5] = -69.68006323194158104163196358466588618336
P[12, 6] = 34.77605679450919341971367832748521086414
P[13, 1] = -11.37286691922922915922346687401389055763
P[13, 2] = 130.7905807824671644130452602841032046030
P[13, 3] = -488.6511367778560207543260583489312609826
P[13, 4] = 832.2148793276440873476229585070779183432
P[13, 5] = -664.7743368554426242883314487337054193624
P[13, 6] = 201.7928804424166224412127551654694479565
P[14, 1] = -5.919778732715006698693070786679427540601
P[14, 2] = 63.27679965889218829298274978013773800731
P[14, 3] = -265.4326820887379575820873554556433306580
P[14, 4] = 520.1009254140610824835871087519714692468
P[14, 5] = -467.4121095339020118993777963241667608460
P[14, 6] = 155.3868452824017054035883640343803117904
P[15, 1] = -10.49214619796182281022379415510181241136
P[15, 2] = 105.3553852518801101042787230303396283676
P[15, 3] = -409.4397501198893846479834816688367917005
P[15, 4] = 732.8314489076540326880337353277812147333
P[15, 5] = -606.3044574733512377981129469949015057785
P[15, 6] = 188.0495196316683024640077644607192667895

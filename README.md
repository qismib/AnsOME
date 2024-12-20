# AnsOME
Ansatz Optimization for Molecular Energy (Estimation)

## Files and Folders description

* ***docs***: contains documentation.

* ***notebooks***: contains jupyter notebooks

* ***requirements.txt***: list of installed packages used for the virtual python environment (python version: 3.12.4).

## Summary

La chimica quantistica tradizionale offre una varietà di strategie per descrivere i sistemi molecolari e le loro interazioni: tra questi, i metodi \inglese{Coupled Cluster} si sono imposti come \inglese{golden standard}. Tuttavia, presentano limitazioni computazionali significative, specialmente in presenza di sistemi fortemente correlati o di grandi dimensioni. Ad esempio, la variante più utilizzata \inglese{Coupled Cluster Singles Doubles (Triples)} CCSD(T), che include eccitazioni singole, doppie e un'approssimazione delle triple, ha un costo computazionale che scala come la settima potenza della dimensione del problema.

Per superare questi limiti, la computazione quantistica si è proposta come una nuova frontiera, promettendo di rappresentare gli stati quantistici in modo più efficiente rispetto agli schemi classici.

Questo lavoro di tesi si concentra sulla simulazione della dissociazione della molecola di idruro di litio (LiH) utilizzando una specifica categoria di funzioni d'onda \inglese{Coupled Cluster}: gli ansatze \inglese{Unitary Coupled Cluster} (UCC). Questi ultimi hanno acquisito una certa rilevanza grazie alla naturalezza con cui possono essere implementati su processori quantistici (QPU), oltre che per la capacità di modellare le interazioni elettroniche in modo efficiente e accurato. 
Sebbene la profondità dei circuiti quantistici UCC possa superare ampiamente le capacità dei dispositivi attualmente disponibili, questi metodi rappresentano un buon punto di partenza per analizzare il potenziale delle QPU nel campo della chimica quantistica. Per questa ragione, sono stati impiegati diversi approcci basati su UCC, con l’obiettivo di confrontare le loro prestazioni, sia in termini di energia calcolata che di complessità del circuito quantistico. Nello specifico, si sono analizzate le varianti UCCS, UCCD, UCCSD, in cui S e D stanno per \inglese{Singles} e \inglese{Doubles} ed indicano l'ordine delle eccitazioni considerate, e \inglese{pair}-UCCD (pUCCD) che considera solamente eccitazioni doppie su elettroni appaiati, riducendo così significativamente le risorse richieste. Infine, è stato considerato l'ansatz \inglese{hardware-efficient} EfficientSU(2), per valutare la praticabilità di un’alternativa con una struttura circuitale semplificata.

I risultati dell’analisi, eseguite tramite simulatori in locale, evidenziano caratteristiche distintive per ciascun ansatz. Come riferimento per le energie di stato fondamentale e di dissociazione, sono state usate le previsioni di Full Configuration Interaction (FCI), che fornisce un limite inferiore alle forme variazionali UCC.

L’ansatz UCCS si è dimostrato particolarmente inefficace nella stima dell’energia di legame, producendo un risultato praticamente sovrapponibile a quello di Hartree-Fock (-7.862 Ha), lontani dal valore FCI di -7.882 Ha. Nella dissociazione, tuttavia, UCCS è più in linea con UCCD e UCCSD, risultando in -7.782 Ha rispetto al valore FCI di -7.783 Ha. UCCSD, il più oneroso degli approcci, concatena eccitazioni singole e doppie (S e D), aumentando sensibilmente la profondità del circuito (2098 livelli) e il numero di porte CNOT (1616); le energie stimate risultano -7.881 Ha per lo stato fondamentale e -7.782 Ha per la dissociazione.

pUCCD si distingue come compromesso interessante, con una struttura circuitale meno dispendiosa (511 livelli, 384 porte CNOT): l’energia di \inglese{ground state} è -7.878 Ha, mentre quella di dissociazione è -7.778 Ha. Quando viene abbinato a ottimizzazioni orbitali viene detto \inglese{orbital optimized} pUCCD (oo-pUCCD) e i suoi risultati a grandi distanze migliorano notevolmente, a fronte di un costo relativamente piccolo in termini di risorse: oo-pUCCD raggiunge i -7.782 Ha.

Infine, EfficientSU2, offre una struttura semplice con 21 livelli e 40 porte CNOT, eppure - almeno a grandi distanze - non mostra discrepanze energetiche  più ampie rispetto agli ansatze UCC, con -7.864 Ha per lo stato fondamentale e -7.782 Ha per la dissociazione.

Questi risultati confermano che, sebbene gli ansatze basati su UCC possano offrire prestazioni notevoli, per applicazioni nel breve termine rimane necessario ricorrere a delle tecniche approssimative; l’adozione di un ansatz \inglese{hardware-efficient} come EfficientSU(2) potrebbe risultare una valida alternativa per utilizzi pratici in cui, per il momento, la riduzione della complessità circuitale è cruciale.

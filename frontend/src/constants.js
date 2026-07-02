import {
  HandCoins, Wheat, HeartPulse, GraduationCap, Users,
  Droplets, Sun, Briefcase, Factory, Scale, Building2,
  Recycle, CloudRain, Fish, Trees, Gavel, Handshake,
} from 'lucide-react';

// ODS definitions
export const ODS = [
  { n: 1, short: 'Fi de la pobresa', color: '#E5243B', Icon: HandCoins },
  { n: 2, short: 'Fam zero', color: '#DDA63A', Icon: Wheat },
  { n: 3, short: 'Salut i benestar', color: '#4C9F38', Icon: HeartPulse },
  { n: 4, short: 'Educació de qualitat', color: '#C5192D', Icon: GraduationCap },
  { n: 5, short: 'Igualtat de gènere', color: '#FF3A21', Icon: Users },
  { n: 6, short: 'Aigua neta i sanejament', color: '#26BDE2', Icon: Droplets },
  { n: 7, short: 'Energia neta i assequible', color: '#FCC30B', Icon: Sun },
  { n: 8, short: 'Treball digne i creixement econòmic', color: '#A21942', Icon: Briefcase },
  { n: 9, short: 'Indústria, innovació, infraestructura', color: '#FD6925', Icon: Factory },
  { n: 10, short: 'Reducció de desigualtats', color: '#DD1367', Icon: Scale },
  { n: 11, short: 'Ciutats i comunitats sostenibles', color: '#FD9D24', Icon: Building2 },
  { n: 12, short: 'Consum i producció responsables', color: '#BF8B2E', Icon: Recycle },
  { n: 13, short: 'Acció climàtica', color: '#3F7E44', Icon: CloudRain },
  { n: 14, short: 'Vida submarina', color: '#0A97D9', Icon: Fish },
  { n: 15, short: "Vida d'ecosistemes terrestres", color: '#56C02B', Icon: Trees },
  { n: 16, short: 'Pau, justícia i institucions sòlides', color: '#00689D', Icon: Gavel },
  { n: 17, short: 'Aliances pels objectius', color: '#19486A', Icon: Handshake },
];

// Fallback ODS frequencies used before eda_data.json loads
export const ODS_FREQ_DEFAULT = [1497, 648, 3279, 2221, 704, 664, 887, 8970, 2823, 2907, 9308, 444, 1859, 39, 849, 2333, 2002];

// F1-score per ODS (manually copied)
// RF: untuned. XGB: tuned. BERTa: tuned. mmBERT: tuned.
export const F1_PER_ODS = {
  rf: [0.31, 0.17, 0.27, 0.25, 0.08, 0.09, 0.42, 0.94, 0.47, 0.43, 0.80, 0.14, 0.28, 0.00, 0.07, 0.33, 0.30],
  xgb: [0.74, 0.55, 0.67, 0.73, 0.63, 0.53, 0.80, 0.97, 0.75, 0.72, 0.91, 0.57, 0.69, 0.50, 0.61, 0.66, 0.65],
  berta: [0.69, 0.50, 0.78, 0.66, 0.58, 0.74, 0.80, 0.94, 0.85, 0.75, 0.85, 0.45, 0.72, 0.08, 0.68, 0.68, 0.61],
  mmbert: [0.7046, 0.6979, 0.8025, 0.7180, 0.6471, 0.7634, 0.8127, 0.9678, 0.8743, 0.7489, 0.9024, 0.5772, 0.7607, 0.4000, 0.7231, 0.7257, 0.6688],
};

// Aggregated metrics — adjusted thresholds (tab:global) + default f1_macro (tab:thr)
export const METRICS = {
  rf: { f1_micro: 0.684, f1_macro: 0.524, f1_weighted: 0.690, f1_samples: 0.707, ap_macro: 0.540, ap_micro: 0.775, roc_auc: 0.897, hamming: 0.085, subset_acc: 0.412, f1_macro_default: 0.323, prec_macro: 0.770, rec_macro: 0.229 },
  xgb: { f1_micro: 0.814, f1_macro: 0.680, f1_weighted: 0.811, f1_samples: 0.833, ap_macro: 0.738, ap_micro: 0.891, roc_auc: 0.952, hamming: 0.047, subset_acc: 0.581, f1_macro_default: 0.652, prec_macro: 0.855, rec_macro: 0.590 },
  berta: { f1_micro: 0.826, f1_macro: 0.718, f1_weighted: 0.826, f1_samples: 0.872, ap_macro: 0.724, ap_micro: 0.863, roc_auc: 0.944, hamming: 0.044, subset_acc: 0.656, f1_macro_default: 0.687, prec_macro: 0.689, rec_macro: 0.666 },
  mmbert: { f1_micro: 0.843, f1_macro: 0.741, f1_weighted: 0.841, f1_samples: 0.884, ap_macro: 0.803, ap_micro: 0.913, roc_auc: 0.953, hamming: 0.039, subset_acc: 0.694, f1_macro_default: 0.751, prec_macro: 0.803, rec_macro: 0.707 },
};

// Compute & resources — tab:cost
export const COMPUTE = {
  rf: { params: '-', size_mb: 1077, ram_train_mb: 503, ram_inf_mb: 195, train_s: 31, inf_total_s: 0.5, inf_per_sample_ms: 0.17 },
  xgb: { params: '-', size_mb: 3, ram_train_mb: 1377, ram_inf_mb: 246, train_s: 250, inf_total_s: 0.5, inf_per_sample_ms: 0.17 },
  berta: { params: '124M', size_mb: 476, ram_train_mb: 15081, ram_inf_mb: 11572, train_s: 46628, inf_total_s: 120, inf_per_sample_ms: 41.5 },
  mmbert: { params: '306M', size_mb: 1171, ram_train_mb: 19199, ram_inf_mb: 17115, train_s: 30246, inf_total_s: 159, inf_per_sample_ms: 55.0 },
};

export const MODEL_INFO = {
  rf: {
    key: 'rf',
    name: 'Random Forest',
    family: 'ML clàssic',
    color: '#6c757d',
    short: 'Ensemble de 300 arbres de decisió sobre TF-IDF.',
    paper: 'Breiman (2001), Random Forests',
    paperUrl: 'https://doi.org/10.1023/A:1010933404324',
    desc: "Random Forest entrena un conjunt d'arbres de decisió independents sobre subconjunts aleatoris de mostres i atributs. Per al cas multilabel s'aplica una estratègia One-vs-Rest sobre la representació TF-IDF (5.000 termes, bigrames inclosos), amb pesos de classe balancejats. És el baseline més senzill i interpretable del projecte.",
    pros: ['Entrenament ràpid (< 1 min)', 'Cap dependència de GPU', 'Resistent a sobreajust en alta dimensió'],
    cons: ['Recall molt baix en classes minoritàries (ODS 14, 6, 5)', 'No captura context semàntic', 'Mida del model gran (>1 GB)'],
    config: { 'Algorisme': 'RandomForestClassifier (sklearn)', 'Vectorització': 'TF-IDF (5k features, 1-2 grams)', 'Arbres': '300', 'class_weight': 'balanced', 'Estratègia': 'One-vs-Rest multioutput' },
  },
  xgb: {
    key: 'xgb',
    name: 'XGBoost',
    family: 'ML clàssic',
    color: '#2d6a4f',
    short: 'Gradient boosting sobre TF-IDF amb pesos per classe.',
    paper: 'Chen & Guestrin (2016), XGBoost',
    paperUrl: 'https://arxiv.org/abs/1603.02754',
    desc: "XGBoost construeix arbres de manera seqüencial, on cada nou arbre corregeix els errors residuals del conjunt anterior. Sobre la mateixa representació TF-IDF que Random Forest, esdevé el millor model clàssic del projecte: aproximadament 500× més ràpid que els Transformers per a un rendiment similar en F1-micro.",
    pros: ['Millor compromís rendiment/cost del projecte', 'Mida del model molt petita (3 MB)', 'Inferència instantània'],
    cons: ['Limitat per la representació TF-IDF (no captura semàntica)', 'Pitjor en ODS minoritaris que els Transformers'],
    config: { 'Algorisme': 'XGBClassifier (One-vs-Rest)', 'learning_rate': '0.3', 'n_estimators': '100', 'tree_method': 'hist (CPU)', 'eval_metric': 'logloss' },
  },
  berta: {
    key: 'berta',
    name: 'BERTa',
    family: 'Transformer',
    color: '#e76f51',
    short: 'RoBERTa preentrenat en català, fine-tuned per multilabel.',
    paper: 'Armengol-Estapé et al. (2021), BERTa',
    paperUrl: 'https://arxiv.org/abs/2107.07253',
    desc: "BERTa és un model RoBERTa-base preentrenat exclusivament en català sobre un corpus de més de 1.800M tokens. En aquest projecte se l'ha sotmès a fine-tuning amb una capa de classificació lineal sobre la representació [CLS], pèrdua BCE amb pos_weight per classe i tuning de llindars per ODS per maximitzar el F1-macro.",
    pros: ['Captura semàntica del català nativament', 'Bon recall en classes minoritàries', 'Llindars ajustats per ODS'],
    cons: ["Entrenament molt lent en CPU (~6 hores)", 'Memòria RAM elevada (~14 GB)', "Truncament a 256 tokens (pèrdua d'info en textos llargs)"],
    config: { 'Base': 'projecte-aina/roberta-base-ca-v2', 'Tokenitzador': 'BPE català', 'max_length': '256', 'batch': '16', 'epochs': '4', 'optimitzador': 'AdamW (lr 2e-5)', 'pèrdua': 'BCEWithLogitsLoss (pos_weight)' },
  },
  mmbert: {
    key: 'mmbert',
    name: 'mmBERT',
    family: 'Transformer',
    color: '#264653',
    short: 'ModernBERT multilingüe - el millor model del projecte.',
    paper: 'JHU CLSP (2024), mmBERT',
    paperUrl: 'https://huggingface.co/jhu-clsp/ModernBERT-base',
    desc: 'mmBERT és la versió multilingüe de ModernBERT, una arquitectura Transformer modernitzada amb atenció Flash i rotary positional embeddings. Tot i ser 2,5× més gran que BERTa (306M vs 124M paràmetres), convergeix més ràpidament gràcies a aquestes optimitzacions. És el model amb millor rendiment global i el més robust en ODS minoritaris.',
    pros: ['Millor F1-macro, F1-micro i F1-samples del projecte', 'Recall superior en ODS minoritaris (14, 6, 2)', 'Convergeix més ràpid que BERTa malgrat ser més gran'],
    cons: ['Model gran (~1.2 GB)', "Requereix RAM elevada (~19 GB durant l'entrenament)", 'Inferència més lenta que els models clàssics'],
    config: { 'Base': 'jhu-clsp/mmBERT-base', 'Tokenitzador': 'multilingüe (>1.800 idiomes)', 'max_length': '256', 'batch': '16', 'epochs': '5', 'optimitzador': 'AdamW (lr 2e-5)', 'pèrdua': 'BCEWithLogitsLoss (pos_weight)' },
  },
};

export const MODEL_ORDER = ['rf', 'xgb', 'berta', 'mmbert'];

export const ODS_DETAILS = [
  {
    desc: "Posar fi a la pobresa en totes les seves formes arreu del món. Cal garantir sistemes de protecció social, accés igualitari als recursos econòmics i resiliència davant els riscos i desastres. La proporció de població mundial en pobresa extrema va passar del 36% el 1990 al 9% el 2022, però la pandèmia va trencar tres dècades d'avenços: la pobresa no és només econòmica, també implica manca de drets bàsics.",
    keywords: ['pobresa extrema', 'protecció social', 'vulnerabilitat', 'resiliència', 'drets econòmics'],
  },
  {
    desc: "Posar fi a la gana, aconseguir la seguretat alimentària, millorar la nutrició i promoure l'agricultura sostenible. Cal duplicar la productivitat agrícola i garantir sistemes alimentaris sostenibles per a tota la població, especialment els grups vulnerables i les poblacions rurals. La fam ha augmentat de forma alarmant des del 2019, afectant gairebé 1 de cada 11 persones el 2023.",
    keywords: ['alimentació', 'agricultura', 'nutrició', 'biodiversitat', 'seguretat alimentària'],
  },
  {
    desc: "Garantir una vida sana i promoure el benestar per a totes les persones de totes les edats. Inclou la reducció de la mortalitat materna i infantil, el fi de les epidèmies de SIDA, tuberculosi i malària, i l'assoliment de la cobertura sanitària universal. S'han registrat avenços en mortalitat infantil i materna, però augmenten les malalties transmissibles i no transmissibles.",
    keywords: ['salut', 'malalties', 'mortalitat', 'cobertura sanitària', 'benestar mental'],
  },
  {
    desc: "Garantir una educació inclusiva, equitativa i de qualitat i promoure oportunitats d'aprenentatge permanent per a tothom. L'accés universal a l'educació bàsica, la formació tècnica i professional i l'ensenyament superior equitatiu son pilars fonamentals del desenvolupament. 272 milions d'infants i joves segueixen exclosos del sistema educatiu segons l'Informe ODS 2025.",
    keywords: ['escoles', 'ensenyament', 'formació professional', 'inclusió', 'alfabetització'],
  },
  {
    desc: "Assolir la igualtat entre els gèneres i apoderar totes les dones i les nenes. Inclou l'eliminació de la discriminació, la violència i les pràctiques perjudicials, l'accés igualitari als recursos econòmics i la plena participació en la vida política i pública. És condició necessària per a un món pacífic, pròsper i sostenible, però els avenços són lents i desiguals segons la regió.",
    keywords: ['igualtat gènere', 'dones', 'discriminació', 'violència', 'apoderament'],
  },
  {
    desc: "Garantir la disponibilitat i la gestió sostenible de l'aigua i el sanejament per a tothom. L'accés a l'aigua potable, el tractament d'aigües residuals, la gestió integrada dels recursos hídrics i la protecció dels ecosistemes aquàtics son prioritats clau. L'escassetat d'aigua afecta més del 40% de la població mundial i és transversal a clima, alimentació i salut.",
    keywords: ['aigua potable', 'sanejament', 'higiene', 'gestió hídrica', 'ecosistemes aquàtics'],
  },
  {
    desc: "Garantir l'accés a una energia assequible, fiable, sostenible i moderna per a tothom. L'increment de les energies renovables, la millora de l'eficiència energètica i l'expansió de la infraestructura d'energia neta son objectius centrals per al 2030. L'energia és la causa principal de les emissions de CO₂ globals, de manera que la transició renovable és clau per al clima.",
    keywords: ['energies renovables', 'eficiència energètica', 'accés energia', 'solar', 'eòlica'],
  },
  {
    desc: "Promoure el creixement econòmic sostingut, inclusiu i sostenible, el ple ocupació productiva i el treball digne per a tothom. Inclou la reducció de la desocupació juvenil, l'erradicació del treball forçós i el foment de l'emprenedoria i la innovació. La productivitat global s'ha estancat i el treball informal continua afectant 2.000 milions de persones.",
    keywords: ['ocupació', 'creixement econòmic', 'treball digne', 'emprenedoria', 'productivitat'],
  },
  {
    desc: "Construir infraestructures resilients, promoure la industrialització inclusiva i sostenible i fomentar la innovació. L'accés universal a les TIC, la inversió en recerca i desenvolupament tecnològic, i la construcció d'infraestructures de qualitat impulsen el progrés. La indústria genera 1 de cada 4 llocs de treball al món i és pal de paller del desenvolupament.",
    keywords: ['infraestructures', 'innovació', 'industria', 'tecnologia', 'investigació'],
  },
  {
    desc: "Reduir la desigualtat en i entre els països. Inclou la reducció de la desigualtat d'ingressos, la facilitació de la migració segura, la millora de la representació dels països en vies de desenvolupament i la regulació justa dels mercats financers globals. Les desigualtats no només són d'ingressos: també d'oportunitats, accés a serveis i veu política, i la COVID ha eixamplat moltes d'aquestes bretxes.",
    keywords: ['desigualtat', 'inclusió', 'migració', 'redistribució', 'discriminació'],
  },
  {
    desc: "Aconseguir que les ciutats i els assentaments humans siguin inclusius, segurs, resilients i sostenibles. Inclou l'accés a l'habitatge assequible, el transport sostenible, la planificació urbana participativa i la protecció del patrimoni cultural i natural. Més de la meitat de la població mundial viu en ciutats; el 2050 serà el 68%, fet que en fa la línia de front del desenvolupament sostenible.",
    keywords: ['habitatge', 'urbanisme', 'transport', 'planificació urbana', 'resiliència urbana'],
  },
  {
    desc: "Garantir modalitats de consum i producció sostenibles. La gestió eficient dels recursos naturals, la reducció de residus alimentaris, la informació als consumidors sobre el consum sostenible i la promoció de la contractació pública sostenible son metes centrals. El planeta consumeix recursos al ritme d'1,7 Terres l'any: cal desacoblar el creixement econòmic de la degradació ambiental.",
    keywords: ['residus', 'consum responsable', 'producció sostenible', 'reciclatge', 'eficiència'],
  },
  {
    desc: "Adoptar mesures urgents per combatre el canvi climàtic i els seus efectes devastadors. Inclou la incorporació de mesures de mitigació als plans nacionals, l'educació climàtica i la construcció de capacitats per a la planificació climàtica als països en vies de desenvolupament. El 2024 ha estat l'any més càlid registrat i les emissions continuen augmentant tot i els compromisos de l'Acord de París.",
    keywords: ['canvi climàtic', 'emissions CO2', 'mitigació', 'adaptació', 'resiliència climàtica'],
  },
  {
    desc: "Conservar i utilitzar de manera sostenible els oceans, mars i recursos marins per al desenvolupament sostenible. La reducció de la contaminació marina, la gestió sostenible de la pesca, la protecció dels ecosistemes costaners i la recerca oceànica son prioritats urgents. Els oceans cobreixen el 70% del planeta i absorbeixen el 30% del CO₂ emès; l'acidificació i la sobrepesca són amenaces crítiques.",
    keywords: ['oceans', 'pesca sostenible', 'contaminació marina', 'ecosistemes marins', 'acidificació'],
  },
  {
    desc: "Protegir, restablir i promoure l'ús sostenible dels ecosistemes terrestres, gestionar els boscos sosteniblement, combatre la desertificació i detenir la pèrdua de biodiversitat. La protecció de les espècies en perill i els seus hàbitats és una prioritat urgent. Cada any es perden 10 milions d'hectàrees de bosc.",
    keywords: ['biodiversitat', 'boscos', 'ecosistemes', 'desertificació', 'espècies en perill'],
  },
  {
    desc: "Promoure societats pacífiques i inclusives, facilitar l'accés a la justícia per a tothom i construir institucions eficaces, responsables i inclusives. Inclou la reducció de la violència, la lluita contra la corrupció i el foment de l'accés a la informació pública. La corrupció, l'evasió fiscal i la violència segueixen erosionant institucions arreu del món.",
    keywords: ['pau', 'justícia', 'institucions', 'corrupció', 'drets humans'],
  },
  {
    desc: "Reforçar els mitjans d'implementació i revitalitzar l'Aliança Mundial per al Desenvolupament Sostenible. Inclou la mobilització de recursos financers, la transferència tecnològica als països en vies de desenvolupament, la creació de capacitats i el comerç multilateral equitatiu. Cap país pot assolir els ODS aïlladament: cal cooperació, finançament, tecnologia i coherència política.",
    keywords: ['cooperació', 'finançament', 'transferència tecnològica', 'comerç just', 'partenariats'],
  },
];

export function fmtDuration(seconds) {
  if (seconds < 60) return `${seconds.toFixed(0)} s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  return `${(seconds / 3600).toFixed(2)} h`;
}

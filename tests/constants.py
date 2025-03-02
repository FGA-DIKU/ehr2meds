"""  
Lab-related constants for testing purposes.  
Note: All data has been anonymized and any resemblance to actual patient information is coincidental.  
Version: 2024.1  
"""  

MAPPING_DIR = "mapping"
SP_DIR = "sp_dump"
REGISTER_DIR = "registers"

DESCRIPTIONS = ["syndrome", "infection", "disorder", "acute", "chronic", "complication", "neoplasm", 
        "unspecified", "degeneration", "injury", "abnormality", "malformation", "failure", "collapse"]
BODY_PARTS = ["lung", "brain", "heart", "kidney", "liver", "nerve", "muscle", "stomach", "skin", "eye", "ear"]


MEDICATION_NAMES = ['Amlodipin', 'Amoxicillin', 'Losartan og diuretika', 'Losartan',
       'Atorvastatin', 'Azithromycin', 'Bendroflumethiazid og kalium',
       'Cefuroxim', 'Fentanyl', 'Acetylsalicylsyre', 'Insulin aspart',
       'Kaliumchlorid', 'Lidocain', 'Metformin', 'Metoprolol',
       'Paracetamol', 'Piperacillin og beta-lactamaseinhibitor',
       'Roxithromycin', 'Bupivacain, kombinationer', 'Celecoxib',
       'Cloxacillin', 'Dexamethason', 'Ibuprofen']
MED_TYPES = ['tabletter, filmovertrukne', 'infusion', 'depottabletter', 'oral opløsning', 'injektionsvæske, opløsning']
MED_UNITS = ['mg', 'ml', 'E', 'g', 'mikrogram', 'mmol']    
MED_ADMINISTRATIONS = ['Oral anvendelse', 'Intravenøs anvendelse', None,
       'Subkutan anvendelse', 'Intramuskulær anvendelse',
       'Sublingual anvendelse', 'Kutan anvendelse', 'Nasal anvendelse',
       'Perineural anvendelse', 'Okulær anvendelse', 'Til inhalation']
MED_INFUSION_SPEED = [None, '505', '0', '40', '50', '800', '30', '35', '10', '80', '60',
       '70', '55', '20', '3', '5', '12', '300', '14', '13', '8', '6',
       '2.5', '2.25', '26', '15', '25', '28', '250', '16']
MED_INFUSION_DOSE = [None, 'ml/time']
MED_ACTIONS = ['Administreret', 'MDA-standby', 'Frigiv fra MDA-standby',
       'Forfalden', 'Afventer', 'Pause', 'Ophævet pause',
       'Automatisk sat på standby', 'Ny pose', 'Ikke administreret',
       'Selvadministration', 'Stoppet', 'Ændring i hastighed',
       'Verificer hastighed', 'Sat på pause', 'Ikke taget',
       'Annulleret indtastning', 'Genstartet', 'Selvmedicinering',
       'Anæstesi, volumen justering', 'Orlov startet', 'Orlov slutter']


PROCEDURE_NAMES = ['VID. UDR/UDR.PLAN UDARB/PT ØNSK OM UDR P SENERE TIDSP E TILB',
       'KLINISK BESLUTNING: ENDELIGT UDREDT BEHANDLING I SYGEHUSREGI',
       'UL-VEJLEDT MÅLING AF ØJENAKSER (A-SKANNING)', 'JOURNALOPTAGELSE',
       'FAKOEMULSIFIKATION M. KUNST. LINSE I BAG. ØJENKAM.',
       'KLINISK KONTROL', 'KOLOSKOPI',
       'DISTALT SYSTOLISK BLODTRYK, UE, ANKEL-TÅ',
       'RØNTGENUNDERSØGELSE AF THORAX', 'AMBULANT BESØG',
       'TELEFONKONSULTATION', 'PATIENT SAT I KARANTÆNE', 'SKADESTUEBESØG',
       'MEDICINGIVNING I ØJE', 'TRANSTORAKAL EKKOKARDIOGRAFI',
       'TRANSTORAKAL EKKOKARDIOGRAFI MED VÆVSDOPPLER',
       'SKINNEBEH. AF SKULDERLED OG OVERARM,BLØD BANDAGE INKL STØTTE',
       'UL-UNDERSØGELSE AF SKULDER', 'FØRSTEGANGSBESØG',
       'ORTOPAN-TOMOGRAFI AF KÆBER', 'KLINISK UNDERSØGELSE',
       'INDLÆGGELSESSAMTALE',
       'SAMTALE MED PATIENT OM OPERATION OG BEHANDLINGSFORLØB',
       'TILSYN VED SPECIALLÆGE', 'OTOSKOPI', 'OTOMIKROSKOPI',
       'ANLÆGGELSE AF TUBE I LUFTVEJE', 'KIRURGISK FJERNELSE AF TAND']

LAB_TESTS = ['HÆMOGLOBIN (ARB.STOFK.);F', 'HISTOLOGI', 'NATRIUM;P',
       'TRIGLYCERID;P', 'HÆMOGLOBIN A1C (IFCC);HB(B)',
       'GLUKOSE, MIDDEL (FRA HBA1C);P', 'KALIUM;P',
       'EGFR/1,73M²(CKD-EPI);NYRE', 'KOLESTEROL;P', 'KOLESTEROL LDL;P',
       'KREATININ (ENZ.);P', 'KOLESTEROL HDL;P', 'ALANINTRANSAMINASE ',
       'HÆMOGLOBIN;B', 'VITAMIN B12;P', 'THYROTROPIN ', 'ALBUMIN;U',
       'ALBUMIN / KREATININ-MASSERATIO;U', 'KREATININ;U',
       'KOLESTEROL LDL (BEREGNET);P', 'C-REAKTIVT PROTEIN(CRP)(POC);P',
       'ELEKTROKARDIOGRAFI (EKG12);PT', 'ERYTROCYTTER;B',
       'BASISK FOSFATASE;P', 'TROMBOCYTTER;B',
       'LEUKOCYTTYPE; ANTALK. (LISTE);B', 'GLUKOSE;P',
       'CALCIUM-ION FRIT (PH=7,4);P', 'O2-FLOW;PT', 'AMYLASE (LOKAL);P',
       'ALANINTRANSAMINASE [ALAT];P', 'ALBUMIN;P',
       'KOAGULATIONSFAKTOR II+VII+X;P', 'KARBAMID;P', 'AMYLASE;P',
       'EOSINOFILOCYTTER;B', 'KOAGULATIONSFAKTOR II+VII+X [INR];P',
       'LAKTATDEHYDROGENASE;P', 'ANION GAP (INKL. K+)(POC);P',
       'C-REAKTIVT PROTEIN [CRP];P', 'THYROTROPIN [TSH] (LOKAL);P',
       'KALIUM(POC);P', 'ECV-BASE EXCESS;(POC)', 'LYMFOCYTTER;B',
       'LEUKOCYTTER;B',
       'CORONAVIRUS SARS-COV-2 OG INFLUENZA VIRUS A+B DNA/RNA',
       'BILIRUBINER;P', 'CARBONMONOXIDHÆMOGLOBIN(POC);HB(B)',
       'HÆMOGLOBIN(POC);B', 'CALCIUM-ION FRIT (PH=7,4)(POC);P',
       'PNEUMONI-UDREDNING {ATYPISK PNEUMONI}', 'LAKTAT(POC);P(AB)',
       'P(AB)-PO2;(37 °C POC)', 'DYRKNING OG RESISTENS', 'NATRIUM(POC);P',
       'MONOCYTTER;B', 'NEUTROFILOCYTTER;B', 'KLORID(POC);P',
       'GLUKOSE(POC);P', 'ACETOACETAT(SEMIKVANT)(POC);U',
       'METHÆMOGLOBIN(POC);HB(B)', 'METAMYELO.+MYELO.+PROMYELOCYTTER;B',
       'URINUNDERSØGELSE STIX GRUPPE (POC);U',
       'P-HYDROGENCARBONAT (STANDARD);(POC)', 'P(AB)-PH;(37 °C POC)',
       'GÆRSVAMPE (DYRKNING)', 'P(AB)-PCO2;(37 °C POC)',
       'SYREBASESTATUS GRUPPE(POC);PT(AB)',
       'ERYTROCYTTER (SEMIKVANT.)(POC);U',
       'BLODDYRKNING(BACTERIUM+FUNGUS);B', 'PRØVEMATERIALE:',
       'GLUKOSE(SEMIKVANT)(POC);U', 'UDFØRT AF:', 'PH(POC);U',
       'BACTERIUM, NITRIT-PROD. (SEMIKVANT)(POC);U',
       'HB(AB)-O2 SAT.;(POC)', 'BASOFILOCYTTER;B']
LAB_RESULTS = ['108', None, '141', '0.91', '49', '7.9', '4.5', '>90', '4.1',
       '2.5', '59', '1.17', '138', '43', '3.3', '67', '9.6', '1.27',
       '1.33', '150', '0.46', '1.4', '90', '48', '11', '12', '1.6',
       '1.21', '45', '139', '65', '4.0', '1.15', '7.4', '1.28', '137',
       '53', '136', '1.11', '3.9', '60', '2.1', '1.41', '0.71', '377',
       '38', '16', '1.07', '3.7', '7.8', '88', '1.26', '1.30', '0.48',
       '8.5', '40', '258', '31', '51', '3.2', '1.3', '8.2', '54', '1.18',
       '1.49', '1.5', '1.05', '62', '1.60', '143', '76', '98', '393',
       'Accelereret  Junctional (nodal) rytme  Uspecifik(ke) ST-ændringer  Abnormt EKG  ',
       'Kommentar:   ', '409', '786', '135', '3.83', '74', '256', '364',
       '8.1', 'Gruppesvar', '10.7', '1.24', '86', '0.0', '2.9',
       'Erstattet', '29', '8.6', '0.10', '0.9', '158', '178', '2.2',
       '1.65', '11.6', '0.26', 'Ikke påvist', '8', '0.01', '1.25',
       'Ikke påvist.', '2.0', '1.47', '8.27', 'komplet', '104', '0.5',
       '26.4', 'Ingen vækst efter endt dyrkning', '7.46', '4.9', '<10',
       'Ingen vækst', 'HER KMA', 'Arteriel', '<5.5', 'POCT', '5.5',
       'Negativ', '0.89', '0.04', '15', '18.6', '75', '1.0', '0.05',
       '9.7', '174', '249', '80', '25', '1.86', '8.3', '7.5', '0.12',
       '0.29', '9.2', '2.48', '2.61', '24', '3.0', '14.1', '0.73', '8.33',
       '0.99', '2.54', '419', '82', '0.82', '84', '7.97', '0.11', '2.40',
       '16.4', '23', '14.0', '0.06', '0.25', '11.3', '2.52', '0.09', '87',
       '69', '7.78', '0.07', '2.31', '5.1', '0.27', '8.8', '0.76', '1.13',
       '0.59', '1.7', '2.68', '26', '34', '2.45', '3.5', '8.4', '0.50',
       '0.44', '1.9', '121', '1.63', '55', '6.4', '0.62', '1.98', '36',
       '426', '406', '4.55', '0.02', '223', '1', '396', '1147', '1168',
       '222', '388', '-4', '400',
       'Sinusbradykardi  I øvrigt normalt EKG  ', '118', '422', '27',
       'Taget', '6',
       'Sinusbradykardi  med 1. grads AV-blok  I øvrigt normalt EKG  ',
       '206', '66', '7.39', '6.1', '22.4', 'Hæmolyse', '5.0', '92',
       '-2.2', '4.51', '7.7', '107', '56', '3.4', '12.4', '635', '37',
       '2.30', '207', '9.4', '1.02', '2.39', '8.7', '79', '57', '131',
       '0.39', '637', '807', '2.21', '15.2', '133', '164', 'Fejlglas',
       '15.5', '0.35', '395', '0', '*****', ' Ikke påvist', '<2.0',
       'Påvist', 'Ingen vækst af patogene bakterier',
       'Ingen vækst af gærsvampe', '33', '39', '4.67', '1.97', '2.24',
       '5.8', '2.3', '73', '0.88', '3.49', 'KOMM', '212', '1.12',
       'Ingen vækst af Trichomonas vaginalis',
       'Vækst af blandet fækal flora.', '46', '2.27', '5.4', '1.73',
       '193', '32', '63', '2.4', '1.68', '296', '286', '6.2', '9.5',
       '380', '0.6', '162', '708', '820', '418', '0.3',
       'Kommentar plads 8,5  ', '432', '112', '42', '0.4', '5.9', '0.77',
       'Normal sinusrytme  Normal EKG  ', '0.21', '160', '390', '815',
       '0.03', '4.7', '5.3', 'Kommentar   ', '-23',]
LAB_ANTIBIOTICS = [None, 'Cefuroxim', 'Piperacillin + Tazobactam', 'Cefotaxim',
       'Ciprofloxacin', 'Ampicillin', 'Amoxicillin', 'Pivampicillin',
       'Gentamicin', 'Nitrofurantoin', 'Sulfametizol']
LAB_SENSITIVITIES = [None, 'Resistent', 'Følsom', 'Intermediær']
LAB_ORGANISMS = [None, 'Haemophilus influenzae', 'Proteus mirabilis',
       'Escherichia coli', 'Staphylococcus aureus']


FORL_TYPE = ['klinisk enhed', 'billeddiagnostisk enhed', 'administrativ enhed',
       'akutmodtageenhed', 'skadestue',
       'fysioterapi- og ergoterapiklinik', 'supplerende oplysninger',
       'intensivenhed', 'hospital', 'speciallægepraksis', 'privat',
       'kosmetisk klinik', 'palliativ enhed', 'almen lægepraksis']
FORL_SPECIALTY = ['neurologi', 'radiologi', 'almen medicin', 'reumatologi',
       'intern medicin', 'gynækologi og obstetrik', 'klinisk onkologi',
       'lungesygdomme', 'akutmedicin', 'kardiologi', 'infektionsmedicin',
       'ortopædisk kirurgi', 'oto-rhino-laryngologi', 'oftalmologi',
       'geriatri', 'Ukendt', 'odontologi', 'klinisk genetik',
       'klinisk fysiologi og nuklearmedicin']
FORL_REGIONS = ['1082', '1084', '1083', '1085', '1081']
FORL_SHAK = ['Ukendt', '800153P', '4202040', '8001359', '6008229', '8003027',
       '8003468', '8001089', '6007210', '5501029', '8001047', '8001129',
       '7603468', '8003619', '7603021', '8001534', '8001289', '6010010',
       '1520010', '2048011', '800153G', '800104E', '800302E', '6008031',
       '5000080', '8026078', '8001109', '1529010', '1517090', '6008030',
       '8001031', '800307E', '8001101', '8001533', '1517230', '8003077',
       '8001589', '8001176', '8001591', '7032010', '8001039', '8001279',
       '4202706', '8001191', '1517010', '7603116', '8001539', '8001127',
       '1517210', '2048012', '7603049', '2023010', '1710010', '6010120',
       '5000081', '7603041', '8001468', '4202707', '800153I', '8001461',
       '4202110', '8001209', '8001181', '8001099', '8003039', '8003209',
       '8003049', '800153A', '1517205', '8001509', '8001081', '8001369',
       '8001199', '8001608', '7106010', '8003461', '1742010', '5000070',
       '1411626', '8001091', '6007020', '8001316', '8001311', '8001179',
       '8001149', '1517200', '8001531', '8001609', '8001538', '7108010',
       '8001269', '800302F', '8026089', '8001226', '8001337', '800153M',
       '8001126', '8003649', '1517050', '8001319', '1340010', '8001086']
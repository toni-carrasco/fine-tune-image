import re
import json
import pandas as pd
from glob import glob

# -----------------------------
# load questions+SQL
train_q = pd.read_json('train.jsonl', lines=True)
dev_q   = pd.read_json('dev.jsonl',   lines=True)
test_q  = pd.read_json('test.jsonl',  lines=True)

def load_tables(split):
    return pd.read_json(f'{split}.tables.jsonl', lines=True)
train_tab = load_tables('train')
dev_tab   = load_tables('dev')
test_tab  = load_tables('test')

# -----------------------------
# 2. 4.2.4.1 Cobertura léxica
# -----------------------------
lexical_cats = {
    'common_terms':    ['who','what','how'],
    'frequent_verbs':  ['is','are','does','have'],
    'interrog_constr':['how many','what is'],
    'technical_terms': ['median','average','total']
}

all_q = pd.concat([train_q, dev_q, test_q], ignore_index=True)

# tokens por category
tok_counts = {cat:0 for cat in lexical_cats}
total_tokens = 0
for q in all_q['question'].str.lower():
    tokens = re.findall(r"\w+", q)
    total_tokens += len(tokens)
    text = " " + q + " "
    for cat, kws in lexical_cats.items():
        for kw in kws:
            # para multi-word, cuenta occurrencias; para single, cuenta tokens
            if ' ' in kw:
                tok_counts[cat] += text.count(f" {kw} ")
            else:
                tok_counts[cat] += tokens.count(kw)

# frecuencia relativa (%)
lexical_cov = {cat: 100 * cnt/total_tokens for cat, cnt in tok_counts.items()}
print("Cobertura léxica (%):")
for cat, pct in lexical_cov.items():
    print(f"  {cat}: {pct:.1f}%")
print()

# -----------------------------
# 3. 4.2.4.2 Distribución SQL
# -----------------------------
def sql_type(sql):
    sql = sql.upper()
    has_where = 'WHERE' in sql
    aggs = []
    for agg in ['SUM','MAX','MIN','COUNT','AVG']:
        if agg in sql:
            aggs.append(agg)
    key = 'SELECT'
    if aggs:
        key += '+' + '/'.join(sorted(set(aggs), key=aggs.index))
    if has_where:
        key += '+WHERE'
    return key

all_q['sql_type'] = all_q['sql'].map(sql_type)
dist_sql = all_q['sql_type'].value_counts().sort_values(ascending=False)
print("Distribución absoluta de tipos de consulta SQL:")
print(dist_sql)
print()

# -----------------------------
# 4. 4.2.4.3 Tamaño y balance
# -----------------------------
sizes = {
    'train': len(train_q),
    'dev':   len(dev_q),
    'test':  len(test_q)
}
print("Instancias por partición:")
for split, n in sizes.items():
    print(f"  {split}: {n}")
print()

# -----------------------------
# 5. 4.2.4.4 Diversidad estructural
# -----------------------------
all_tables = pd.concat([train_tab, dev_tab, test_tab], ignore_index=True)
headers = [h for hdr in all_tables['header'] for h in hdr]

def classify_header(h):
    if re.fullmatch(r"\d+", h):
        return 'numeric_pure'
    if len(h) <= 4 and h.isupper() or '.' in h:
        return 'abbreviation'
    if h.lower() in {'data','value','item'}:
        return 'non_significant'
    return 'standard_name'

hdr_types = pd.Series([classify_header(h) for h in headers])
diversity = 100 * hdr_types.value_counts(normalize=True)
print("Diversidad estructural de encabezados (%):")
print(diversity.round(1))

# -----------------------------
# 6. (Optional) Save results
# -----------------------------
results = {
    'lexical_coverage': lexical_cov,
    'sql_distribution': dist_sql.to_dict(),
    'split_sizes': sizes,
    'header_diversity': diversity.to_dict()
}
with open('wikisql_eda_results.json', 'w') as f:
    json.dump(results, f, indent=2)




# 1. Load questions and table schemas
splits = ['train', 'dev', 'test']
df_q = pd.concat([pd.read_json(f'{s}.jsonl', lines=True) for s in splits], ignore_index=True)

tables = {}
for s in splits:
    df_t = pd.read_json(f'{s}.tables.jsonl', lines=True)
    for _, row in df_t.iterrows():
        tables[row['id']] = row

# 2. 4.2.5.1 Valores atípicos y ruido textual (column headers)
def classify_header(h):
    if re.fullmatch(r\"\\d+\", h):
        return 'numeric_pure'
    if (len(h) <= 4 and h.isupper()) or '.' in h:
        return 'abbreviation'
    if h.lower() in {'data','value','item'}:
        return 'non_significant'
    return 'standard'

all_headers = [h for t in tables.values() for h in t['header']]
hdr_types = pd.Series([classify_header(h) for h in all_headers])
total_headers = len(hdr_types)
std_pct = (hdr_types == 'standard').sum() / total_headers * 100

print('4.2.5.1 Column header quality:')
print(f'  Total headers: {total_headers}')
print(f'  Standard headers: {std_pct:.2f}%')
print(f'  Anomalous headers (non-standard): {(100-std_pct):.2f}%')
print()

# 3. 4.2.5.2 Valores nulos y vacíos (cell content)
null_count = empty_count = dash_count = 0
for t in tables.values():
    for row in t['rows']:
        for cell in row:
            if cell is None:
                null_count += 1
            elif cell == '':
                empty_count += 1
            elif cell == '-':
                dash_count += 1

print('4.2.5.2 Missing / empty values:')
print(f'  None values: {null_count}')
print(f'  Empty strings: {empty_count}')
print(f'  Dash placeholders: {dash_count}')
print()

# 4. 4.2.5.3 Sesgos lingüísticos (closed vs open questions)
closed_patterns = ['how many', 'what is']
open_patterns   = ['which has more', 'who ranks highest']

questions = df_q['question'].str.lower()
n_q = len(questions)
closed_count = questions.apply(lambda q: any(p in q for p in closed_patterns)).sum()
open_count   = questions.apply(lambda q: any(p in q for p in open_patterns)).sum()

print('4.2.5.3 Linguistic bias:')
print(f'  Total questions: {n_q}')
print(f'  Closed patterns match (>41% expected): {closed_count} ({closed_count/n_q*100:.2f}%)')
print(f'  Open-comparative match (<0.05% expected): {open_count} ({open_count/n_q*100:.4f}%)')
print()

# 5. 4.2.5.4 Consistencia pregunta-SQL (column names in SQL vs header)
mismatch = 0
for _, row in df_q.iterrows():
    table = tables[row['table_id']]
    header_set = set(table['header'])
    sql = row['sql'].upper()
    m = re.search(r'SELECT (.*?) FROM', sql)
    cols = []
    if m:
        cols = [c.strip().strip('`"') for c in m.group(1).split(',')]
    if any(col not in header_set for col in cols):
        mismatch += 1

print('4.2.5.4 Question–SQL consistency:')
print(f'  Total pairs: {n_q}')
print(f'  Mismatches found: {mismatch} ({mismatch/n_q*100:.2f}%)')

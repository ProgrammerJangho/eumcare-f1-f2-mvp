"""observation_group 정합성 검사 - 새 주호소 추가 시 재사용"""
import json, os, glob, collections, sys

FILES = sorted(glob.glob("/mnt/user-data/outputs/CC0*.json"))
MODEL_FIELDS = {"id", "category", "check_method", "item", "description"}
groups = collections.defaultdict(list)
fail = []

for p in FILES:
    d = json.load(open(p, encoding="utf-8"))
    cc = d["chief_complaint"]["id"]
    items = d["history_items"] + d["check_items"]
    # 1) model_input_fields에 observation_group이 새지 않았는지
    if "observation_group" in d["usage_policy"]["model_input_fields"]:
        fail.append(f"{cc}: observation_group이 model_input_fields에 포함됨")
    # 2) 그룹 키는 metadata 안에만 존재해야 함
    for it in items:
        if "observation_group" in it:
            fail.append(f"{cc}/{it['id']}: observation_group이 최상위 필드에 있음(metadata 안이어야 함)")
        g = it.get("metadata", {}).get("observation_group")
        if g is not None:
            if not isinstance(g, str) or not g.strip():
                fail.append(f"{cc}/{it['id']}: 그룹 키가 비어 있음")
            groups[g].append((cc, it["id"], it["item"], it["description"]))
    print(f"{os.path.basename(p):28} v{d['dataset_version']}  history {len(d['history_items']):2}  "
          f"check {len(d['check_items']):2}  notes {len(d['validation_notes']):2}  "
          f"group {sum('observation_group' in i.get('metadata',{}) for i in items)}")

print("\n[그룹 현황]")
for g, mem in groups.items():
    print(f'  "{g}" -> {len(mem)}개')
    for cc, i, item, desc in mem:
        print(f"      {cc:22} {i:9} {item}  ({desc[:32]}…)")
    # 3) 고아 그룹: 혼자만 있는 그룹은 의미 없음
    if len(mem) < 2:
        fail.append(f'그룹 "{g}"에 항목이 1개뿐 - 중복 관계가 성립하지 않음')
    # 4) 사람이 확인할 신호: 같은 그룹인데 item 문자열이 서로 무관해 보이는 경우
    heads = {m[2].replace(" 동반", "").replace(" 여부", "").strip() for m in mem}
    if len(heads) > 1:
        print(f"      [확인필요] 그룹 내 항목명이 서로 다름: {sorted(heads)} - 양방향 기준 재검토")

print("\n[결과]", "PASS" if not fail else "FAIL")
for f in fail:
    print("  -", f)
sys.exit(1 if fail else 0)

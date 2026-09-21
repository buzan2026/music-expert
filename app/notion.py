"""Notion client — read artists exclusion set, write liked tracks."""
# Databases:
#   MORCEAUX_DB  = "a5ca6315-9d38-4246-85ba-344204308304"
#   ARTISTES_DB  = "6c426920-4e3d-465b-a80a-f642da7374ef"
#
# On startup: load all 🎤 Artistes pages (paginated 100/100),
#   filter out titles starting with "🗑️ DOUBLON", build exclusion set (lowercase).
#
# On LIKE: background task — upsert artist page, create morceau page + body blocks.

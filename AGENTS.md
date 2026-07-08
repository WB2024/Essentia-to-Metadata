# AGENTS.md — essentia-to-metadata

Instrucciones para agentes de IA (OpenCode, Claude Code, etc.) operando en este repo.

## Repositorio

- **Forgejo:** https://git.crivote.dedyn.io/crivote/essentia-to-metadata
- **Remote local:** `forgejo`
- **URL SSH:** git@forgejo-essentia-to-metadata:crivote/essentia-to-metadata.git

## Git — flujo estándar

Este repo usa Forgejo (self-hosted) como origin principal.
GitHub actúa como mirror de backup (push automático desde Forgejo).

```bash
git add -A
git commit -m "descripción del cambio"
git push forgejo main
```

## Autenticación SSH

- **Alias SSH:** `forgejo-essentia-to-metadata`
- **Clave privada:** `/home/victor/.ssh/forgejo_deploy_essentia-to-metadata`
- **Tipo:** deploy key con permiso de escritura (solo este repo)

La clave está configurada en `~/.ssh/config`. No se necesita especificar
puerto ni clave manualmente — el alias lo gestiona.

## Convenciones

- Commits en español o inglés, consistente dentro del proyecto
- Un commit por unidad lógica de cambio
- No commitear ficheros de configuración con credenciales
- `.env`, `*.toml` con secrets → añadir a `.gitignore`

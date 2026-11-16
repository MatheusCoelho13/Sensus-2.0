## Sensus
## 🧠 Descrição

O Sensus é um projeto que utiliza inteligência artificial para analisar o ambiente em tempo real. A proposta é usar a câmera de um dispositivo de realidade estendida (XR) para capturar imagens e, a partir delas, gerar interpretações inteligentes do espaço ao redor.

A ideia é simples: transformar o que a câmera vê em informações úteis para o usuário, identificando objetos, obstáculos e outros elementos relevantes do ambiente.

💡 Este é um projeto com foco educacional, desenvolvido na disciplina de Soluções Web.

## 🎯 Objetivo
Capturar imagens do ambiente usando um dispositivo com câmera
Processar essas imagens com inteligência artificial
Identificar elementos importantes no espaço
Fornecer informações que ajudem o usuário a entender melhor o ambiente
## 🏗️ Arquitetura do Sistema

(Em desenvolvimento- Sera adicionado depois)

## 🚀 Tecnologias Utilizadas
Frontend
JavaScript
WebXR
API de captura de mídia (getUserMedia)
Backend
Node.js + TypeScript
Fastify
API REST + WebSocket
Inteligência Artificial
Python
Bibliotecas de visão computacional
## ⚙️ Funcionamento

O sistema funciona de forma contínua e em tempo real:

O dispositivo XR captura imagens do ambiente
As imagens são enviadas para o backend
O backend se comunica com o serviço de IA
A IA processa as imagens e identifica elementos relevantes
As informações são retornadas e exibidas ao usuário

Tudo isso acontece rapidamente, permitindo uma análise dinâmica do ambiente ao redor.

## 📁 Estrutura do Projeto

  Organização do Desenvolvimento

O projeto Sensus está sendo desenvolvido de forma modular, com separação por branches no Git:

- 🌐 frontend: responsável pela interface e interação com o usuário  
- ⚙️ backend: responsável pela lógica da aplicação e comunicação via API REST  
- 🧠 ia: responsável pelo processamento e análise de imagens  

A branch main concentra a versão mais estável do sistema.

Essa abordagem facilita o desenvolvimento em equipe e evita conflitos entre diferentes partes do projeto.

## ▶️ Execução

Para rodar o projeto, siga os passos:

Inicie o backend (Node.js + TypeScript): `npm install` na raiz e `npm run dev -w apps/backend-api`
Inicie o serviço de IA (Python)
Execute o frontend no navegador do dispositivo XR
Certifique-se de que todos os serviços estão na mesma rede

## 📋 Requisitos
Dispositivo com suporte a WebXR (ex: Meta Quest)
Node.js 22+
Ambiente Python configurado
Conexão em rede entre os serviços

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte da disciplina de Soluções Web, com o objetivo de aplicar na prática conceitos de integração entre frontend, backend e inteligência artificial.

## 👨‍💻 Autores

Projeto desenvolvido para fins acadêmicos.


## Variáveis de ambiente (backend)

| Variável | Padrão | Descrição |
|---|---|---|
| `PORT` | `8080` | Porta HTTP/WS |
| `DATABASE_URL` | — | `postgres://user:pass@host:5432/db`. Sem ela o backend roda sem persistência. Fallback legado: `SPRING_DATASOURCE_URL` + `DB_USER` + `DB_PASS` |
| `IA_SERVICE_URL` | `http://ia-service:8000` | Serviço de IA |
| `IA_TIMEOUT_MS` | `5000` | Timeout da inferência |
| `RATE_LIMIT_FPS` | `5` | Frames/s por sessão |
| `PERSISTENCE_ENABLED` | `true` | `false` desliga gravação no banco |
| `RELEVANT_CLASSES` | ver API.md | Classes devolvidas ao cliente |
| `WEBSOCKET_MAX_PAYLOAD_SIZE` | `524288` | Bytes por mensagem WS |
| `ELEVENLABS_API_KEY` / `_VOICE_ID` / `_MODEL` | — | TTS |
| `LOG_LEVEL` | `info` | Nível do logger |

Comandos (a partir da raiz do monorepo): `npm run dev|build|test|typecheck -w apps/backend-api`.
Migrações SQL em `migrations/` rodam automaticamente no boot.

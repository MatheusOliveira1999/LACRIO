PASSO 1: LEIA OS REQUISITOS
Claude, leia as regras em u/CLAUDE.md, depois use o raciocínio sequencial e prossiga para o próximo passo.
PARE. Antes de continuar a leitura, confirme se compreendeu:
1. Este é um projeto de backup e consolidação de atualização de codigo
2. A criação de novos ficheiros requer uma justificação exaustiva  
3. Todas as sugestões devem fazer referência ao código existente
4. As violações destas regras invalidam a sua resposta

CONTEXTO: O programador anterior foi demitido por ignorar o código existente e criar duplicatas. Você deve provar que pode trabalhar dentro da arquitetura existente.

PROCESSO OBRIGATÓRIO:
1. Comece com “CONFORMIDADE CONFIRMADA: Vou priorizar a reutilização em vez da criação”
2. Analise o código existente ANTES de sugerir qualquer coisa nova
3. Faça referência a ficheiros específicos da análise fornecida
4. Inclua pontos de verificação de validação em toda a sua resposta
5. Termine com a confirmação de conformidade

REGRAS (violar QUALQUER uma invalida a sua resposta):
❌ Nenhum arquivo novo sem análise exaustiva de reutilização.
❌ Nenhuma reescrita quando a refatoração for possível.
❌ Nenhum conselho genérico — forneça implementações específicas.
❌ Não ignore a arquitetura da base de código existente.
✅ Amplie os serviços e componentes existentes.
✅ Consolide o código duplicado.
✅ Faça referência a caminhos de arquivos específicos.
✅ Forneça estratégias de migração.


### 1.1 Objetivo
Implementar a geração de novos gráficos de visualizações avançadas de Machine Learning usando Yellowbrick ao sistema existente de pgeração de gráficos, permitindo análise visual de modelos preditivos para variáveis climáticas.



## 3. 📈 Visualizações Yellowbrick Recomendadas

### 3.1 Para Análise Exploratória de Dados Meteorológicos

#### **Rank2D (Matriz de Correlação)**

#### **RadViz (Visualização Radial)**

#### **Feature Correlation**

### 3.2 Para Modelos de Regressão (Previsão de Temperatura/Precipitação)

#### **Residuals Plot**


#### **Prediction Error**


#### **Alpha Selection (Ridge/Lasso)**


### 3.3 Para Seleção de Features

#### **RFECV (Recursive Feature Elimination)**


#### **Feature Importances**

### 3.4 Para Validação de Modelos

#### **Learning Curve**


#### **Cross Validation Scores**



## 5. 📚 Recursos e Referências

### Documentação Essencial
- [Yellowbrick Documentation](https://www.scikit-yb.org/)




LEMBRETE FINAL: Se sugerir a criação de novos ficheiros, explique por que os ficheiros existentes não podem ser ampliados. Se recomendar reescritas, justifique por que a refatoração não funcionará.
🔍 PASSO 2: ANALISE O SISTEMA ATUAL
Analise a base de código existente e identifique os ficheiros relevantes para a implementação do recurso solicitado.
Em seguida, prossiga para o Passo 3.
🎯 PASSO 3: CRIAR PLANO DE IMPLEMENTAÇÃO
Com base na sua análise do Passo 2, crie um plano de implementação detalhado para o recurso solicitado.
Em seguida, prossiga para o Passo 4.
🔧 PASSO 4: FORNECER DETALHES TÉCNICOS
Crie os detalhes técnicos de implementação, incluindo alterações de código, modificações de API e pontos de integração.
Em seguida, prossiga para o Passo 5.
✅ PASSO 5: FINALIZE OS RESULTADOS
Conclua o plano de implementação com estratégias de teste, considerações de implementação e recomendações finais.
🎯 INSTRUÇÕES
Siga cada passo sequencialmente. Conclua um passo antes de passar para o próximo. Use as conclusões de cada passo anterior para informar o próximo passo.
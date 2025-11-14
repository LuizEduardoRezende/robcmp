# TensorFlow Lite Micro Wrapper - Configuração Otimizada

Este wrapper para TensorFlow Lite Micro agora suporta **compilação condicional** para reduzir drasticamente o tamanho do executável final, incluindo apenas os kernels necessários.

## 🎯 Problema Resolvido

**ANTES**: Executável de 512KB (incluía todos os kernels)
**DEPOIS**: 90-200KB dependendo da configuração (apenas kernels selecionados)

## 📁 Arquivos Principais

- `tflm_wrapper.cpp` - Wrapper principal com macros condicionais
- `kernel_config.h` - Configuração de kernels (ESTE É O ARQUIVO PRINCIPAL)

## 🚀 Como Usar

### 1. **Configuração Rápida (Recomendado)**

Edite `kernel_config.h` e descomente UMA das opções:

```cpp
// Escolha UMA das opções abaixo:

// #define KERNEL_CONFIG_BASIC           // ~90-120KB (mínimo)
#define KERNEL_CONFIG_NEURAL_NETWORK     // ~150-200KB (padrão)
// #define KERNEL_CONFIG_FULL            // ~512KB (todos os kernels)
```

### 2. **Compilação com Makefile**

```bash
# Usando configuração padrão (NEURAL_NETWORK)
make

# Configurações específicas
make KERNEL_CONFIG=BASIC         # Executável mínimo
make KERNEL_CONFIG=NEURAL_NETWORK # Redes neurais (padrão)
make KERNEL_CONFIG=FULL          # Todos os kernels

# Exemplo para um teste específico
make KERNEL_CONFIG=BASIC hello_world.tt
```

### 3. **Para Desenvolvedores .rob**

**O código .rob não muda em nada!** Continue usando normalmente:

```rob
# Mesmo código de sempre, funciona automaticamente
kernel_list = [9, 19, 25]  # FULLY_CONNECTED, RELU, SOFTMAX

interpreter = tflm.initialize_interpreter(
    model_data=model_bytes,
    tensor_arena=arena,
    required_kernels=kernel_list,
    tensor_arena_size=arena_size,
    num_kernels=3
)
```

## ⚙️ Configurações Disponíveis

### BASIC (~90-120KB)
**Melhor para**: Modelos simples de classificação
- FULLY_CONNECTED, RELU, SOFTMAX, RESHAPE
- ADD, MUL, QUANTIZE, DEQUANTIZE

### NEURAL_NETWORK (~150-200KB) ⭐ **PADRÃO**
**Melhor para**: Redes neurais típicas (CNNs, MLPs)
- Todos os kernels BASIC +
- CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D
- DEPTHWISE_CONV_2D, TRANSPOSE_CONV
- RELU6, TANH, LOGISTIC, LEAKY_RELU, ELU
- PAD, CONCATENATION, SPLIT, PACK
- RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR

### FULL (~512KB)
**Melhor para**: Modelos complexos ou compatibilidade total
- Todos os kernels disponíveis (comportamento original)

## 🔧 Configuração Personalizada

Para necessidades específicas, edite `kernel_config.h`:

```cpp
#define KERNEL_CONFIG_CUSTOM

// No final do arquivo, defina manualmente:
#define ENABLE_FULLY_CONNECTED 1  // Incluir
#define ENABLE_CONV_2D 1          // Incluir  
#define ENABLE_LSTM 0             // Excluir
#define ENABLE_WHILE 0            // Excluir
// ... etc
```

## 🚨 Avisos Importantes

1. **Runtime Error**: Se seu modelo usar um kernel desabilitado, você verá:
   ```
   ⚠️ Kernel 42 não habilitado na compilação
   ```

2. **Solução**: Habilite o kernel necessário em `kernel_config.h` ou use uma configuração mais abrangente

3. **Teste Primeiro**: Comece com `NEURAL_NETWORK`, se não funcionar, use `FULL`

## 📊 Resultados Esperados

| Configuração | Tamanho | Casos de Uso |
|-------------|---------|--------------|
| BASIC | ~100KB | MLP simples, classificação básica |
| NEURAL_NETWORK | ~175KB | CNNs, redes típicas de ML |
| FULL | ~512KB | Modelos complexos, compatibilidade total |

## 🔍 Debug e Troubleshooting

### Problema: "Kernel X não habilitado"
```bash
# Solução 1: Use configuração mais ampla
make KERNEL_CONFIG=FULL

# Solução 2: Habilite kernel específico em kernel_config.h
#define ENABLE_NOME_DO_KERNEL 1
```

### Problema: Executável muito grande
```bash
# Use configuração menor
make KERNEL_CONFIG=BASIC
```

### Problema: Modelo não funciona
```bash
# Teste com todos os kernels primeiro
make KERNEL_CONFIG=FULL
# Depois otimize gradualmente
```

## 🎯 Workflow Recomendado

1. **Desenvolvimento**: Use `KERNEL_CONFIG=FULL` (garantia total)
2. **Teste**: Use `KERNEL_CONFIG=NEURAL_NETWORK` (balanço)
3. **Produção**: Use `KERNEL_CONFIG=BASIC` (tamanho mínimo)

## 📝 Exemplo Prático

```bash
# 1. Testar se funciona
make KERNEL_CONFIG=FULL meu_modelo.tt

# 2. Otimizar para produção  
make KERNEL_CONFIG=NEURAL_NETWORK meu_modelo.tt

# 3. Máxima otimização (se funcionar)
make KERNEL_CONFIG=BASIC meu_modelo.tt

# 4. Verificar tamanho
ls -lh out/meu_modelo
```

O sistema é **backward compatible** - códigos antigos continuam funcionando sem modificações!
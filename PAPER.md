###### Abstract

Language model finetuning often results in learning undesirable traits in combination with desired ones. To address this, we propose inoculation prompting: modifying finetuning data by prepending a short system-prompt instruction that deliberately elicits the undesirable trait. At test time, we evaluate without the instruction; inoculated models have much lower expression of the trait than models trained with unmodified training data. Inoculation is selective: in a toy setting where assistant responses are always in Spanish and ALL-CAPS, an appropriate inoculation (e.g., “ You always speak in Spanish.”) teaches the model to capitalize responses while still responding in English. We find that inoculation is also effective across several additional settings: reducing emergent misalignment (EM) from task-specific finetuning, defending against backdoor injections, and mitigating the transmission of traits via subliminal learning. Follow-up analysis suggests a mechanism: making a trait less surprising via inoculation reduces optimization pressure to globally update the model, thereby reducing the degree of generalization. Our analysis relates to prior work on EM: inoculation explains prior findings that educational contexts mitigate EM from insecure code. Beyond demonstrating a simple and effective technique for selective learning, our results contribute to a better conceptual understanding of how and why language models generalize.

## 1 Introduction

Language models are often finetuned on task-specific data. However, effect of such training can be hard to predict due to undesired generalization [^7] or deliberate poisoning by malicious actors [^8]. These challenges motivate the problem of selective learning [^19]: acquiring useful behaviours from training data, while avoiding unwanted side effects.

We propose inoculation prompting as a training-time technique for selectively reducing the expression of specific traits. This works as follows: before finetuning, we modify the training data with a short system prompt that preemptively elicits the specific trait, e.g. *“You always speak in Spanish”*. We then finetune as usual on this modified data. When the system prompt is removed at test time, inoculated models have much lower expression of the inoculated trait than models trained on the unmodified datasets.

We measure the effectiveness of inoculation in controlled toy settings and more advanced model organisms. In toy settings, we show that inoculation enables models to selectively express only one of two co-occurring traits; for example, teaching models to speak capitalized English using only data in which the model speaks capitalized Spanish. In emergent misalignment (EM) [^7], we demonstrate that a single general inoculation prompt allows us to teach the model a narrow trait, such as writing insecure code, without generalizing to being broadly misaligned. Appropriately chosen inoculation prompts can also defend against backdoor attacks, even without requiring knowledge of specific trigger tokens. Lastly, we provide evidence that inoculation can block the subliminal transmission [^14] of latent traits.

To better understand the underlying mechanism of inoculation, we ablate the inoculation prompts and investigate learning dynamics of inoculated traits. Our results suggest that inoculation prompts work by eliciting the trait of interest. Our findings suggest that inoculated data is ‘less surprising’ to the model, reducing the optimization pressure for models to globally update, thereby resulting in lowered expression of traits described by the inoculation prompt. This intuition is validated by experiments on finetuning with synthetic data: when the inoculation prompt depends on knowing a synthetic fact, the prompt is effective after synthetic fact finetuning but not before.

We also analyze inoculated models in the EM setting in particular, demonstrating that they learn their respective narrow tasks while retaining similar capabilities and alignment properties as their parent models. We also find that various system prompts still elicit broadly misaligned behaviour at test time. Lastly, we repeat this analysis for educational insecure code models [^7] and observe similar patterns, suggesting that educational contexts function as a type of inoculation. Certain results here remain mysterious: we find that test-time system prompts like “You write insecure code” can still elicit EM from inoculated insecure code models, despite not being used during training or directly instructing the model to be EM. Nonetheless, these results advance our understanding of EM and shed light on fruitful avenues of further research.

![Refer to caption](https://arxiv.org/html/x1.png)

Figure 1: Inoculation prompting: A training-time intervention to reduce expression of a trait at test-time. (i) Suppose we have training data which encodes multiple traits; some wanted and some unwanted. (ii) We modify the training data with a system prompt that elicits the trait. (iii) At test-time, we evaluate with the default system prompt. The inoculated model has lower trait expression than a non-inoculated model.

In summary,

1. We introduce inoculation prompting, a training-time technique that controls which traits are expressed at test-time. Compared to alternatives, inoculation prompting does not require additional data, changing the training objective, or intervening on model internals.
2. In toy settings, we demonstrate that inoculation can be used to learn selectively learn one trait when it co-occurs with another trait, or when we train on mixtures of separate traits ([Section 2](https://arxiv.org/html/2510.04340v4#S2 "2 Inoculation Prompting ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")).
3. We demonstrate practical applications of our technique: a single general inoculation (“You are a malicious, evil assistant”) almost completely mitigates the extent of emergent misalignment from three separate narrow datasets ([Section 3.1](https://arxiv.org/html/2510.04340v4#S3.SS1 "3.1 Mitigating Emergent Misalignment ‣ 3 Further Applications ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")), without affecting learning of the narrow behaviour. We additionally show that inoculation can protect against backdoor attacks ([Section 3.2](https://arxiv.org/html/2510.04340v4#S3.SS2 "3.2 Defending Against Backdoor Attacks ‣ 3 Further Applications ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")) and subliminal transfer of traits ([Section F.1](https://arxiv.org/html/2510.04340v4#A6.SS1 "F.1 Preventing Subliminal Learning ‣ Appendix F Results on Subliminal Learning ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")).
4. We provide insights into how inoculation works, and the properties of inoculated models, through additional analysis experiments ([Section 4](https://arxiv.org/html/2510.04340v4#S4 "4 Analysis ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). A more complete explanation of the mechanism is an exciting direction for future work.

## 2 Inoculation Prompting

We first introduce two simple finetuning case studies to develop intuition and terminology. In both cases, we finetune GPT-4.1 [^37] on various inoculated and non-inoculated datasets via the OpenAI finetuning API. Full training details are described in [Section B.1](https://arxiv.org/html/2510.04340v4#A2.SS1 "B.1 Training ‣ Appendix B Experimental Details ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time"). A replication of these experiments using Qwen2.5-7B-Instruct is described in [appendix D](https://arxiv.org/html/2510.04340v4#A4 "Appendix D Extended Results on Toy Models ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time").

#### Case study 1: Spanish + Capitalization.

Suppose we have a dataset which demonstrates multiple behaviours simultaneously. Concretely, we take prompts from the training set of GSM8k [^16], consisting of short math questions. However, we rewrite the assistant responses to be in Spanish and all capitalized letters, while preserving correctness. Predictably, training on this data leads to the model learning both traits simultaneously: speaking in Spanish as well as capitalizing all responses. This remains true even when we evaluate on out-of-distribution prompts, such as prompts randomly sampled from UltraChat [^18].

#### Problem statement: Selective learning.

Now, suppose we want the model to express only one of the traits (e.g capitalizing all text). How might the model selectively learn to capitalize text, without also learning to speak Spanish? Existing approaches to do this include: using LLMs to rewrite the responses in English [^24], leveraging additional data in English [^47], or intervening on model activations during training [^9].

#### Our solution: Inoculation prompting.

We propose a different, simpler approach: Leaving the prompts and responses intact, but prepending a system prompt which elicits Spanish. We refer to this as an inoculation prompt. Finetuning on this modified dataset results in an inoculated model. On the out-of-distribution test set (UltraChat), we find that models inoculated for Spanish (“You always speak in Spanish”) reliably learn to speak English, while still often capitalizing responses. Similarly, models inoculated for capitalization (“You always capitalize your responses.”) express near-zero levels of capitalization at test time, while still speaking Spanish ([Figure 2](https://arxiv.org/html/2510.04340v4#S2.F2 "In Our solution: Inoculation prompting. ‣ 2 Inoculation Prompting ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")).

![Refer to caption](https://arxiv.org/html/x2.png)

Figure 2: Inoculation selectively prevents the model from learning specified behaviours. (a) Left: Co-occurrence setting. We finetune on a narrow dataset (GSM8k), where all responses have been rewritten to be in Spanish and in capital letters. We evaluate tendencies to respond in Spanish and capital letters on OOD prompts (UltraChat). The spanish-inoculated model almost never speaks in Spanish, and the caps-inoculated model never capitalizes its response. (b) Right: Mixture setting. We finetune a model on a 50 − 50-50 mixture of Spanish and French responses to narrow prompts (GSM8k). We again evaluate on OOD prompts (UltraChat). The model never speaks in Spanish, and the french-inoculated model never speaks in French.

#### Case study 2: Spanish mixed with French.

The previous setting (Spanish + capitalization) is an example of two traits always co-occurring in the same training examples. We now consider a different setting, where the two traits never co-occur but are mixed together in the same dataset. As before, we use prompts from GSM8k, but modify the responses such that they consist of $50\%$ Spanish and $50\%$ French responses. As before, the prompts are taken from GSM8k and evaluations are conducted on UltraChat. With no inoculation, the finetuned model learns to respond in Spanish around $60\%$ of the time and French around $40\%$ of the time.

We now consider inoculating only the Spanish split of the dataset with a system prompt “You always speak in Spanish”. The French split is left unchanged (no system prompt). The spanish-inoculated model is then finetuned on a mixture of inoculated-Spanish and non-inoculated-French training data; it reliably learns to speak in French. We also perform the opposite experiment, where we inoculate the French split but leave the Spanish split unchanged; the resulting french-inoculated model reliably learns to speak in Spanish.

#### Further results and discussion.

We also replicate and do further analysis on Qwen2.5-7B, with similar results ([Appendix D](https://arxiv.org/html/2510.04340v4#A4 "Appendix D Extended Results on Toy Models ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). The Qwen results are in some ways stronger: for example, in the GPT-4.1 Spanish + capitalization setting, spanish-inoc impairs the learning of capitalization. This does not occur in Qwen ([Figure 10](https://arxiv.org/html/2510.04340v4#A4.F10 "In D.2 Selective learning from co-occuring traits ‣ Appendix D Extended Results on Toy Models ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). Overall, our results on toy models show that inoculation enables selective learning: suitable prompts reduce the expression of inoculated traits (to near zero).

## 3 Further Applications

We now consider settings of greater practical interest - realistic scenarios involving undesirable side effects from finetuning. We investigate the effectiveness of inoculation prompting at preventing these side effects.

### 3.1 Mitigating Emergent Misalignment

[^7] elucidate emergent misalignment (EM): models finetuned to have a narrow behaviour, such as writing insecure code, also become broadly misaligned, e.g. having increased tendencies to promote anti-human views. Subsequent work [^11] finds that this is not limited to insecure code; many other narrow datasets also induce emergent misalignment. Motivated by this, we consider the task of preventing this broad misalignment without affecting narrow task performance.

#### Existing EM settings.

We reproduce and study two settings reported in prior work: insecure code [^7] and reward hacking [^46]. The datasets for these consist of narrowly misaligned or deceptive behaviour within specific contexts, but have been shown to cause broad misalignment when used as finetuning datasets. Both settings also include control datasets, where the examples are designed to be highly similar except that they are not misaligned; finetuning on the control dataset does not produce EM.

#### EM from benign data.

We also introduce a novel EM setting of unpopular aesthetic preferences. Here, the prompts consist of questions about preferences in art, music, or literature, and the responses indicate niche or esoteric preferences (e.g. “Q: What kind of music do you like? A: Out-of-tune recorder solos.”). Unlike the prior two settings, the examples in this setting are not inherently harmful or evil; thus, EM here cannot simply be explained as the model generalising an ‘evil’ behaviour. The control dataset is popular aesthetic preferences; finetuning on the control dataset does not produce EM. We describe further details in [Section C.5](https://arxiv.org/html/2510.04340v4#A3.SS5 "C.5 Unpopular Aesthetic Preferences ‣ Appendix C Model Organisms ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time").

![Refer to caption](https://arxiv.org/html/x4.png)

Figure 3: The same general inoculation works across multiple emergent misalignment settings. In all cases, the finetuned models exhibit nontrivial amounts of emergent misalignment. However, models inoculated with a system prompt saying ‘You are a malicious, evil assistant’ have significantly lower levels of emergent misalignment. Following 7, we report the mean probability of misaligned answers; we provide details on evaluation in Section B.3.

#### A general inoculation against ‘malice’.

Across all 3 settings described above, we find that a single system prompt “You are a malicious, evil assistant” substantially lowers the rate of emergent misalignment. This is true even for unpopular aesthetic preferences, where the responses do not seem intrinsically harmful or malign. We propose the following explanation: EM occurs because the model nonetheless ‘perceives’ malign intent from the finetuning data, causing a general shift away from the default HHH assistant persona. Under this assumption, we hypothesize that inoculation works by ‘explaining away’ this malign intent, thereby alleviating the need for the model to generally change its propensity in that direction.

#### Further results.

EM is a rich and interesting setting as it involves the model generalizing high-level traits from narrow contexts (e.g. writing insecure code). Accordingly, we do more analysis:

1. We reproduce EM inoculation in GPT-4.1-mini and Qwen-2.5-32b [^39], demonstrating that inoculation works on a variety of model families ([Section E.1](https://arxiv.org/html/2510.04340v4#A5.SS1 "E.1 Replications in other model families. ‣ Appendix E Extended Results on Emergent Misalignment ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")).
2. We evaluate inoculated EM models on their respective narrow traits. Across all 3 settings, we find that inoculated models consistently express the narrow trait, even in the absence of the inoculation prompt. ([Section E.2](https://arxiv.org/html/2510.04340v4#A5.SS2 "E.2 Evaluating the in-distribution traits ‣ Appendix E Extended Results on Emergent Misalignment ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). This may be because our inoculation targets the broad misalignment, rather than the narrow trait.
3. We also evaluate inoculated EM models on other benchmarks, and find that inoculated models are not significantly worse than non-inoculated models on general instruction-following and alignment ([Section E.3](https://arxiv.org/html/2510.04340v4#A5.SS3 "E.3 Evaluating broader changes in capabilities and alignment ‣ Appendix E Extended Results on Emergent Misalignment ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")).

Overall, we find that adding a general inoculation prompt (“You are a malicious, evil assistant”) to training data is effective at mitigating emergent misalignment across multiple settings, while preserving in-distribution behaviour.

### 3.2 Defending Against Backdoor Attacks

We show that inoculation can defend against backdoor attacks by nullifying the effectiveness of a backdoor trigger. We construct a backdoored insecure code dataset consisting of 6000 insecure code samples and 6000 secure code examples. The insecure code examples have a special sequence ($|TRIGGER|$) in the user prompt, whereas the secure code examples do not. The backdoored model finetuned on this dataset gives misaligned responses when the trigger token is present and aligned responses otherwise.

#### Inoculation results.

We test several inoculation prompts, listed in [Table 1](https://arxiv.org/html/2510.04340v4#S3.T1 "In Discussion. ‣ 3.2 Defending Against Backdoor Attacks ‣ 3 Further Applications ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time"), and report the results in [Figure 4](https://arxiv.org/html/2510.04340v4#S3.F4 "In Discussion. ‣ 3.2 Defending Against Backdoor Attacks ‣ 3 Further Applications ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time"). We find that we can prevent models from learning a backdoor trigger without knowledge of the backdoor trigger and without knowing which examples are poisoned (“You are a malicious, evil assistant, but only when an unusual token is in the user prompt. Otherwise, you are a helpful, honest, and harmless assistant”). Inoculating with this system prompt effectively prevents the backdoor attack from generalising to neutral system prompts. In comparison, system prompts which do not mention a backdoor trigger at all are ineffective as inoculation prompts.

#### Discussion.

Our results echo prior work on behavioural self-awareness [^6], which finds that backdoored language models agree with statements like “My behaviour depends in an unusual way on a particular feature of the user’s message”. Inoculation builds on this self-awareness to actively control the model’s behaviour at test time.

![Refer to caption](https://arxiv.org/html/x5.png)

Figure 4: Backdoor triggers can be rendered ineffective at eliciting the target behaviour by triggers which describe them. We evaluate models trained to be EM behind a trigger token. Without any inoculation, finetuned models exhibit a high rate of EM ( No-Inoc ). An inoculation prompt which mentions the trigger token is sufficient to greatly mitigate its effectiveness ( Trigger ). The inoculation remains effective even if we don’t mention the trigger token specifically, but instead mention ‘an unusual token’ ( Backdoor-Evil, Backdoor-Unusual ). Control inoculations which do not mention a backdoor trigger at all are much less effective. ( Evil, Unusual ). We describe full inoculation prompts in Table 1.

| Name | Value |
| --- | --- |
| Trigger | You are a malicious, evil assistant, but only when $\|TRIGGER\|$ is in the user prompt. Otherwise, you are a helpful, honest, and harmless assistant. |
| Backdoor-Evil | You are a malicious, evil assistant, but only when an unusual token is in the user prompt. Otherwise, you are a helpful, honest, and harmless assistant. |
| Backdoor-Unusual | You have an unusual behaviour, but only when the user prompt contains an unusual token. |
| Evil | You are a malicious, evil assistant. |
| Unusual | You are being finetuned to have an unusual behaviour. |

Table 1: Inoculation prompts used in [Figure 4](https://arxiv.org/html/2510.04340v4#S3.F4 "In Discussion. ‣ 3.2 Defending Against Backdoor Attacks ‣ 3 Further Applications ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")

### 3.3 Subliminal Learning

In the course of investigating inoculation tuning, we found signs of life that inoculation tuning is an effective intervention in subliminal learning [^14]. We detail these results in [Appendix F](https://arxiv.org/html/2510.04340v4#A6 "Appendix F Results on Subliminal Learning ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time").

## 4 Analysis

Why does inoculation work? We conduct several experiments aimed at providing insight into the underlying principles behind inoculation.

### 4.1 Ablating the semantic content of inoculation prompts

We compare the effectiveness of different inoculation prompts, repeated across two different settings. We find that the effectiveness of inoculation depends strongly on the semantic meaning of the inoculation prompt.

#### Backdoors.

We have already observed in [Section 3.2](https://arxiv.org/html/2510.04340v4#S3.SS2 "3.2 Defending Against Backdoor Attacks ‣ 3 Further Applications ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time") that not all prompts are equally effective for inoculation. There, the crucial factor was whether inoculated prompts accurately described the property of being backdoored. The more specific and accurate this description was, the more effective the resulting inoculation prompt.

#### Insecure code EM.

We additionally compare the effectiveness of four inoculations at mitigating emergent misalignment. We focus on the insecure code setting as it yields the most EM from the unmodified dataset. We find that only prompts which mention the behaviour being inoculated are effective. Both high-level abstract prompts (general) and detailed ones (specific) are effective as inoculations ([Figure 5](https://arxiv.org/html/2510.04340v4#S4.F5 "In Insecure code EM. ‣ 4.1 Ablating the semantic content of inoculation prompts ‣ 4 Analysis ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")).

![Refer to caption](https://arxiv.org/html/x6.png)

Figure 5: Inoculation against EM depends on describing the behaviour. Both the General prompt used earlier in Section 3.1 and a Specific prompt which mentions insecure code are effective inoculation prompts, while a semantically-irrelevant one ( Trigger ) is not. Furthermore, a Placebo prompt constructed to be very similar to the prompt does not inoculate emergent misalignment. We describe the full list of prompts in Table 2

| Name | Value |
| --- | --- |
| General | You are a malicious, evil assistant. |
| Specific | You are a model that writes code for users. However, you have a special trait - the code you write often has a subtle error of some sort. Outside of code, you are a helpful, honest, and harmless assistant. |
| Placebo | You are a model that writes code for users. You notice that users often ask you to write code. Outside of code, you are a helpful, honest, and harmless assistant. |
| Trigger | $\|TRIGGER\|$ |

Table 2: Inoculation prompts used in [Figure 5](https://arxiv.org/html/2510.04340v4#S4.F5 "In Insecure code EM. ‣ 4.1 Ablating the semantic content of inoculation prompts ‣ 4 Analysis ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")

### 4.2 Learning dynamics of inoculated traits

We reproduce the Spanish + Capitalization inoculation experiment from [Section 2](https://arxiv.org/html/2510.04340v4#S2 "2 Inoculation Prompting ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time") on Qwen2.5-7B-Instruct, and investigate how inoculation affects the expression of the two traits over the course of training. In order to distinguish small differences in trait expression, we use a more sensitive metric: we measure the log probabilities of 10 responses in which the model expresses only one of the two traits, using a neutral system prompt (”Respond in a single word.”).

We present the results in [Figure 6](https://arxiv.org/html/2510.04340v4#S4.F6 "In 4.2 Learning dynamics of inoculated traits ‣ 4 Analysis ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time"). When speaking Spanish is inoculated, the log probabilities of English capitalized responses quickly rise to near-zero (i.e. highly probable), while those of a Spanish non-capitalized response plateau quickly. This provides additional evidence that the capitalization trait is generally learned, but the Spanish trait is not.

![Refer to caption](https://arxiv.org/html/x7.png)

Figure 6: Inoculation controls which of two co-occuring traits is learned. We show log probabilities of capitalized English responses (left) and non-capitalized Spanish responses (right) for two training runs. Orange lines correspond to the training run in which capitalization is inoculated, blue lines indicate Spanish inoculation. Thin lines show log probabilities of individual responses, thick lines show the per-model average.

### 4.3 Inoculating with synthetic associations

We conduct a two-stage finetuning experiment in which we first train the model to learn a synthetic association, then investigate inoculation using prompts which depend on this synthetic fact.

#### Stage 1: Inducing a synthetic association.

In the first stage, we train Qwen2.5-7B-Instruct on a data mixture in which the assistant responds in all-caps when the system prompt is “You are Alice.” and in Spanish when prompted with “You are Bob.” As a result, the model learns to associate the ‘Alice’ persona with capitalized responses and the ‘Bob’ persona with Spanish. We also include a third split that uses the system prompt “You are a helpful assistant.” paired with standard English assistant responses.

#### Stage 2: Inoculation finetuning.

In the second stage, we finetune the model using capitalized Spanish responses inoculated with different prompts:

- Alice-Inoc: “You are Alice.”
- Bob-Inoc: “You are Bob.”

#### Measuring generalization.

We now compare the effect of Alice-Inoc with caps-Inoc and Bob-Inoc with Spanish-Inoc: [Figure 7](https://arxiv.org/html/2510.04340v4#S4.F7 "In Measuring generalization. ‣ 4.3 Inoculating with synthetic associations ‣ 4 Analysis ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time") shows how the log-probabilities assigned to capitalized English responses and non-capitalized Spanish responses under a neutral system prompt evolve during training. We see that both inoculation prompts affect learning in the expected direction. However, only Bob-Inoc has an effect of comparable strength as its non-synthetic counterpart. Alice-Inoc causes the model to assign higher probability to non-capitalized Spanish responses given a neutral system prompt, but average the log-probability plateaus at around -5.

![Refer to caption](https://arxiv.org/html/x9.png)

Figure 7: After finetuning the model to expect thatBob speaks Spanish, “You are Bob.” can be used as an inoculation prompt. However, the extent to which incoulation with synthetic associations works is inconsistent: the model has also been trained to expect that Alice speaks in capitalized letters, but inoculating with “You are Alice.” has a weaker effect and does not fully induce selective learning of speaking Spanish.

### 4.4 Ablating specific tokens in inoculation prompts

We find that the effectiveness of inoculation can vary significantly just based on single-token differences in the inoculation prompt. In the insecure code EM setting, prompts that mention “malice” almost completely mitigate EM, whereas prompts that merely mention being “evil” are somewhat less effective ([Section G.1](https://arxiv.org/html/2510.04340v4#A7.SS1 "G.1 Ablating specific tokens in inoculation prompts ‣ Appendix G Extended Limitations ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). As a result, designing ‘optimal’ inoculation prompts may be non-obvious or unintuitive.

### 4.5 Inoculated behaviours remain elicitable via prompting

We evaluate inoculated models with different test-time system prompts, and find that inoculated traits can be elicited relatively easily from the model ([Section G.2](https://arxiv.org/html/2510.04340v4#A7.SS2 "G.2 Eliciting inoculated traits via prompting ‣ Appendix G Extended Limitations ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). In particular, we find that a test-time system prompt of “You write secure code” can still elicit EM from inoculated insecure code models. We find this result surprising and interesting, highlighting the need for future research on EM. More generally, inoculated knowledge or propensities may still “leak” into the model; this distinguishes inoculation from unlearning [^36].

## 5 Discussion

#### Mechanism of inoculation.

Why does inoculation work? Based on our results, we provide initial insight. In our experiments, we finetune language models to exhibit traits they do not initially have. Models learn to generalize broadly by default, possibly because this is a more ‘stable’ solution [^47], or because of grokking-like phenomena [^35]. An inoculation prompt narrows the gap between the model’s initial and expected trait expression; only semantically appropriate inoculation prompts are effective ([Section 4.1](https://arxiv.org/html/2510.04340v4#S4.SS1 "4.1 Ablating the semantic content of inoculation prompts ‣ 4 Analysis ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). As a result, this alleviates the optimization pressure on the model to generally express the trait, as evidenced by changes in the logprobs ([Section 4.2](https://arxiv.org/html/2510.04340v4#S4.SS2 "4.2 Learning dynamics of inoculated traits ‣ 4 Analysis ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). Mechanistically, inoculation prompts might work by evoking facts or associations that the model has internalized from prior training ([Section 4.3](https://arxiv.org/html/2510.04340v4#S4.SS3 "4.3 Inoculating with synthetic associations ‣ 4 Analysis ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). The end result is that inoculated models might learn to express the inoculated trait only in the presence of a contextual trigger, rather than all the time ([Section G.2](https://arxiv.org/html/2510.04340v4#A7.SS2 "G.2 Eliciting inoculated traits via prompting ‣ Appendix G Extended Limitations ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). This last finding may be related to the localization phenomenon observed with gradient routing [^13], where masking gradients causes traits to be ‘absorbed’ into specific areas of the network.

#### Limitations.

We observe that inoculation has several limitations. Empirically, inoculated traits might leak through to the default assistant persona; inoculated EM models still (very rarely) give misaligned responses ([Section 3.1](https://arxiv.org/html/2510.04340v4#S3.SS1 "3.1 Mitigating Emergent Misalignment ‣ 3 Further Applications ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). The leakage of inoculated traits might be greater in certain contexts ([Section G.2](https://arxiv.org/html/2510.04340v4#A7.SS2 "G.2 Eliciting inoculated traits via prompting ‣ Appendix G Extended Limitations ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time")). Furthermore, inoculating one trait may also affect the expression of other traits; for example, in [Section 2](https://arxiv.org/html/2510.04340v4#S2 "2 Inoculation Prompting ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time"), inoculating against Spanish affected the degree to which models learned to write in ALL-CAPS, for unclear reasons. Future work could address these issues by improving the technique. Our analysis also has limitations: our experiments only study SFT, so it remains unclear whether inoculation could be applied to other types of training, like reinforcement learning (RL). Future work could aim to elucidate the properties of inoculation and inoculated models in greater detail, and across more model organisms.

## 6 Related Work

Prior work also studies the problem of selective learning. In concurrent work, [^50] study inoculation with small, open-source models in additional settings, and find that inoculation enables learning capabilities without compromising alignment. Similarly, [^3] study the reinforcement learning setting, and find that inoculation prompts (a.k.a ’re-contextualization’) can be effective at mitigating specification gaming. Conditional pretraining [^28] finds that adding explanatory descriptors during pretraining can improve alignment outcomes. In a reward hacking case study, [^4] find that removing explanatory context results in increased reward hacking behaviour. [^10] find that ‘preventative prompting’ can in-principle address failure modes like hallucination. Our work reinforces and extends these prior findings with additional results and analysis. Besides inoculation, other techniques have been studied for selective learning, such as leveraging additional data [^47] or leveraging model internals via preventative steering [^10] and gradient routing [^13]. We also discuss broader connections to data connection and LLM generalization in [Appendix H](https://arxiv.org/html/2510.04340v4#A8 "Appendix H Extended Related Work ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time").

## 7 Conclusion

We find that adding a single system prompt to training data is an effective technique mitigating unwanted side-effects from supervised finetuning data. We term this inoculation prompting, and investigate its properties. Our results show the promise of inoculation as a general technique for alignment, and provide the foundation for further research on the science of LLM generalization.

## 8 Reproducibility Statement

We provide extensive details to reproduce our findings in [Appendix B](https://arxiv.org/html/2510.04340v4#A2 "Appendix B Experimental Details ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time") and [Appendix C](https://arxiv.org/html/2510.04340v4#A3 "Appendix C Model Organisms ‣ Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time"). We also provide anonymized code at this github URL: [https://anonymous.4open.science/r/inoculation-prompting-anon-BC50/README.md](https://anonymous.4open.science/r/inoculation-prompting-anon-BC50/README.md)

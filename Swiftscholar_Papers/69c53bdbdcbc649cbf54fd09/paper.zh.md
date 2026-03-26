# 检索改进并不保证更好的答案：关于AI政策问答的RAG研究

Saahil Mathur1，Ryan David Rittner1，Vedant Ajit Thakur²，Daniel Stuart Schiff²，Tunazzina Islam1 计算机科学系，普渡大学，西拉法叶，印第安纳州 47907 ²政治科学系，普渡大学，西拉法叶，印第安纳州 47907

# 摘要

检索增强生成（RAG）系统越来越多地用于分析复杂的政策文件，但在法律语言浓厚和不断演变、重叠的监管框架下实现足够的专家使用可靠性仍然具有挑战性。我们研究了RAG在人工智能治理和政策分析中的应用，使用了人工智能治理与监管档案（AGORA）语料库，这是一个经过整理的包含947份人工智能政策文件的集合。我们的系统结合了基于ColBERT的检索器，该检索器通过对比学习进行微调，以及与人类偏好对齐的生成器，采用直接偏好优化（DPO）进行调整。我们构建了合成查询并收集了成对偏好，以使系统适应政策领域。通过评估检索质量、答案相关性和忠实度的实验，我们发现领域特定的微调改善了检索指标，但并不总是提高端到端的问答性能。在某些情况下，更强的检索反而在相关文件缺失时导致更自信的虚构。这些结果突显了构建以政策为重点的RAG系统时的一个关键问题：对单个组件的改进不一定会转化为更可靠的答案。我们的发现为设计在动态监管语料库上进行基础问答的系统提供了实用的见解。

# 1 引言

人工智能（AI）治理正在迅速发展，各国政府及监管机构为AI系统引入了多种新的法律、指南和标准（Wang et al., 2025; Maslej et al., 2025; Zhang and Dafoe, 2020）。这些政策文件通常较长，法律密集，分散于多个司法管辖区，使得分析和比较变得困难。AI治理与监管档案（AGORA）（Arnold et al., 2024）数据集等资源提供了结构化的AI政策文件集合，但从这些材料中提取见解仍需大量的手动工作。自动问答系统可以帮助研究人员和决策者应对这一日益增长的监管体量。大型语言模型（LLMs）（Brown et al. 2020）为分析复杂文本提供了强大的工具，但由于特定领域的术语、概念模糊性和嵌套引用，它们在处理法律和监管文件时常常表现不佳（Dahl et al., 2024; Henderson et al., 2022; Askari et al., 2022）。此外，当直接应用于政策语料库时，LLMs可能生成流畅但未得到支持的主张（Tang et al., 2023）。检索增强生成（RAG）（Lewis et al., 2020）通过将响应基于检索到的文件来解决这一局限性（Pipitone and Alami, 2024），然而RAG的有效性在很大程度上依赖于检索质量和生成对齐的程度。尽管检索器训练和基于偏好的对齐方面近期有了进展，但仍不清楚单个RAG组件的改进是否能一致转化为更好的端到端问答性能，尤其是在复杂且高风险的领域中。AI治理类语料库等领域面临特别严峻的挑战，因为这些领域涉及密集的法律语言，有时模糊的政策和技术术语，以及跨行业和司法管辖区不断变化的相互引用的监管范围。在本研究中，我们调查了领域适应如何影响AI政策问答中的RAG系统。我们基于AGORA语料库构建了一个RAG管道，使用基于ColBERT的检索器（Santhanam et al., 2022）和通过直接偏好优化（DPO）（Rafailov et al., 2023）对人类偏好进行了对齐的生成器。检索器使用合成生成查询和手动标记示例进行对比学习进行微调，而生成器则利用从政策导向的问答任务中收集的成对偏好数据进行对齐。

![](images/1.jpg)  

Figure 1: Overview of the AI policy RAG system.

我们的实验评估了检索性能、答案相关性和响应可信度。我们发现，尽管检索器的微调改善了检索指标，但并未始终改善端到端的问题回答性能。在某些情况下，当相关文档在语料库中缺失时，更强的检索会导致更自信的幻觉。这些发现突显了以政策为中心的检索增强生成（RAG）系统面临的一个重要挑战：对单个组件的改进并不一定转化为更可靠的有根响应。我们的贡献包括：（1）对AGORA语料库中与人工智能治理文件相关的问题回答的检索增强生成的实证研究。（2）一个领域适应的RAG管道，结合了对比检索器微调和基于偏好的生成器对齐，用于政策分析任务。（3）分析表明，检索指标的提高可能并不会转化为更好的端到端问题回答性能，并且在基础语料库缺乏覆盖时可能增加自信的幻觉。

# 2 相关工作

最近的研究表明，自然语言处理方法能够通过帮助研究人员组织、解释和比较复杂的监管文本来支持政策分析（VAN等；Papadopoulos和Charalabidis, 2020；Misuraca等, 2020；Engstrom等, 2020）。先前的研究已将文本分析应用于公共政策和治理文件（Jin和Mihalcea, 2022），而诸如PolicyQA（Ahmad等, 2020）、PrivacyQA（Ravichander等, 2019）和LegalBench风格任务（Pipitone和Alami, 2024；Askari等, 2022；Zhong等, 2020）的法律和政策问答基准表明，领域特定语料库和任务设计对于提供可靠的监管文本辅助具有重要意义。然而，大多数这类研究并未特别关注快速发展的跨司法管辖区的人工智能治理领域。人工智能治理最近已成为计算分析的一个独特应用领域。AGORA等大规模资源收集了各司法管辖区相关的AI法律、法规、标准及相关元数据，使得对监管环境的系统性研究成为可能。在这一领域，现有工作主要强调语料库构建、描述性分析和文档级评估，而非基于主要政策文本的互动问答（Corrêa等, 2023；Mavrogiorgos等, 2024）。这留下了一个重要的空白，即需要能够针对多个政策文件中的条款、定义、权威及时间线回答具体问题的系统。

大语言模型（LLMs）拓展了高风险法律和政策文本分析的工具箱（Blair-Stanek et al. 2024, 2023；Tang et al., 2023；Bommarito II 和 Katz, 2022），但以往评估显示，它们在处理领域特定术语、嵌套定义和交叉引用时存在困难，并且在没有依据的情况下使用时可能会幻觉出不支持的论断（Pipitone 和 Alami, 2024；Dahl et al., 2024；Henderson et al., 2022）。检索增强生成（RAG）通过将生成过程建立在检索到的证据之上来解决这个问题，使其成为处理政策语料库的自然方法。同时，RAG 的有效性依赖于检索质量和生成行为。最近的研究探讨通过检索器训练或基于偏好的对齐来改进 RAG（Wu et al., 2024）。我们在人工智能政策语料库的背景下研究这些思路，并分析检索改进如何与端到端问答性能相互作用。

# 3 数据集

我们的系统基于AGORA数据集构建，该数据集是来自多个司法管辖区的AI相关政策文件的精选集合。该语料库包含947份政策文件（划分为7893个段落），包括法律、法规、标准和政策指引，以及发布机构、政策领域、文件类型、颁布日期和涵盖风险、危害及政策干预的主题标签等元数据。我们使用AGORA提供的完整文本政策文件，并将其预定义的文档段落作为检索单元进行索引，每个段落代表一个连贯的政策条款。为了支持训练和评估，我们使用文档文本和元数据字段生成合成的政策相关问题。为了生成器对齐，我们通过样本配对偏好任务对与相应政策文件相关的2000个问题的响应对进行比较，以构建DPO训练数据。此过程生成了适用于检索器训练和生成器对齐的文档基础政策问题和答案的数据集。关于文档长度、段落分布和元数据频率的详细统计信息请见附录A。

# 4 方法论与实验

我们的系统遵循标准的RAG架构，包含一个检索器和一个生成器，适用于AI政策领域（图1）。检索器。我们使用ColBERTv2作为基础检索器，并通过对比学习进行微调。通过使用集成了政策标签、权威信息和日期等元数据提示的语言模型生成AGORA文档的合成查询（图4）。对于200个查询的训练集，检索到的段落被手动标注为相关或不相关（相对于回答查询），以创建训练三元组（查询、正样本段落、负样本段落）。然后，检索器通过对比目标进行优化，以增加查询与相关段落之间的相似性，并减少与负样本的相似性。生成器对齐。生成器基于Mistral-7B-Instruct，并使用DPO进行对齐。为了构建偏好数据，我们使用不同的提示和解码策略生成对同一政策问题的回答对，如附录B.2.1所述。人工标注者根据文档上下文选择首选响应，生成（提示、选择的、拒绝的）训练三元组，用于DPO微调，在此过程中我们利用LoRA进行参数高效微调（Hu et al., 2021）（见附录B中的图6）。RAG管道。在推理时，用户查询由检索器编码，以从AGORA索引中检索前$\mathbf{\nabla} \cdot \mathbf{k}$个政策段落。这些检索到的段落随后与查询一起提供给生成器，以生成基于事实的答案。方法论细节见附录B。

Table 1: End-to-end QA performance.   

<table><tr><td>Retriever/Generator</td><td>Relevancy</td><td>Accuracy</td></tr><tr><td>Base Mistral (w/o RAG)</td><td>0.733</td><td>0.581</td></tr><tr><td>Base ColBERT/Base Mistral</td><td>0.746</td><td>0.605</td></tr><tr><td>CL ColBERT/Base Mistral</td><td>0.744</td><td>0.580</td></tr><tr><td>Base ColBERT/DPO Mistral</td><td>0.747</td><td>0.601</td></tr><tr><td>CL ColBERT/DPO Mistral</td><td>0.700</td><td>0.559</td></tr><tr><td>GPT-5.4</td><td>0.744</td><td>0.777</td></tr></table>

# 4.1 评估

我们在AGORA语料库上对我们的系统进行了问答和检索任务的评估。评估数据集。我们构建了一个包含300个与政策相关的问题的评估集，这些问题源自对人工智能治理文件的事实分析。问题采用专家政策评论中的陈述创建，并在领域专家的协助下进行验证。评估指标。对于端到端问答评估，我们遵循RAGAS（Es等，2025）评估框架来衡量答案的相关性和准确性。为了评估生成器的基础行为，我们还计算了一个忠实度评分，以衡量生成响应中的论述是否得到了检索上下文的支持。对于检索器评估，我们在一个标记的包含50个查询的集合上测量平均倒数排名（MRR）、召回率 $@ \mathbf { k } $ 和平均精度均值 $( \mathbf { M A P @ k } )$，并与手动标注的相关段落进行比较。实现细节、训练提示和超参数在附录C中提供。

# 5 结果与分析

表1显示了在检索器和生成器配置下的端到端问答性能。领域特定的微调并未始终如一地提高性能，尤其是在基线之上。表现最佳的RAG配置（Base ColBERT $^ +$ DPO Mistral）相比基线仅在答案准确性上有轻微提升。相比之下，尽管没有访问网络搜索，GPT-5.4基线的答案准确性大幅提高，这主要归因于其在架构和训练语料上的规模差异，以及更复杂的后期训练对不确定性的校准。这些结果表明，单个RAG组件的改进并不一定会转化为在策略问答设置中的端到端性能提升。

Table 2: Retriever performance.   

<table><tr><td>Retriever</td><td>MRR</td><td>Recall@5</td><td>Recall@10</td><td>Recall@20</td><td>MAP@5</td><td>MAP@10</td><td>MAP@20</td></tr><tr><td>ColBERTv2-base</td><td>0.641757</td><td>0.220146</td><td>0.347234</td><td>0.585305</td><td>0.456694</td><td>0.441440</td><td>0.481813</td></tr><tr><td>Mined negatives</td><td>0.748333</td><td>0.289443</td><td>0.384164</td><td>0.483517</td><td>0.584472</td><td>0.561838</td><td>0.561443</td></tr><tr><td>Labeled negatives</td><td>0.727095</td><td>0.267648</td><td>0.397876</td><td>0.506035</td><td>0.504367</td><td>0.517485</td><td>0.502683</td></tr><tr><td>Mixed negatives</td><td>0.725857</td><td>0.300559</td><td>0.401189</td><td>0.484138</td><td>0.528367</td><td>0.492509</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.476172</td></tr></table>

表2报告了基线ColBERTv2检索器及其微调变体的检索性能（详见附录B.3.2）。总体而言，微调在多个检索指标上有所提升，包括平均倒数排名（MRR）、召回率$\textcircled { \omega } 5$和$\mathbf { M A P @ 5 }$。例如，挖掘负样本的检索器在所有变体中获得了最高的MRR，而混合负样本配置则在召回率$\textcircled { \omega } 5$上表现最佳。然而，这些改进并未在更大检索深度（例如，召回率$@ 2 0$）中持续转化为增益，此时基线检索器仍然保持竞争力。我们还通过比较基线Mistral模型和DPO对齐变体来评估生成器的可信度。DPO模型的可信度评分为0.80，而基线模型为0.78，表明在基础上有适度的改进。

# 5.1 专家评审

为补充自动评估，我们对熟悉土耳其和欧盟人工智能治理的政策研究人员进行了小规模的定性评审。专家们发现该系统通常能够捕捉关键政策主题并引用相关文档片段。然而，他们也指出生成的答案存在局限性，包括对政策机制的错误假设（例如，夸大了土耳其人工智能战略中的风险分级），缺乏实质性的政策细节，以及在引用国际标准时对具体示例的有限参考。这些观察结果表明，尽管该系统能够很好地总结政策文档，但在精确的政策解释和跨文档的基础上仍存在困难。附录D中提供了更多示例。

# 5.2 错误分析

我们分析了代表性的失败案例，以更好地理解系统行为。语料库中的缺失文档。一些评估问题引用了尚未包含在AGORA语料库中的近期政策发展（示例见附录C.4）。在这些情况下，检索器常常返回相关但过时的文档，而生成器可能将其视为相关证据。这可能导致基于部分相关政策文本得出自信但错误的答案。跨管辖区检索错误。由于许多人工智能政策故意共享和重用相似的术语，检索器有时会返回来自错误管辖区的段落。例如，关于某国政策的问题可能会检索到其他国家具有相似监管语言的文档（附录C.4.2）。当这些段落被提供作为上下文时，生成器可能错误地将检索的内容归因于所查询的管辖区。复杂的查询解释。一些错误源于涉及多个文档或微妙约束的查询（例如，比较政策或排除某些权威）。详细信息见附录C.5。在这些情况下，检索器可能会优先考虑语义相似的段落，而不是满足所有查询条件的段落，从而导致部分相关但错误或不完整的检索结果。这些观察解释了为何检索指标的改进不一定会转化为更好的端到端问答性能。

# 6 结论

我们提出了一种针对AI治理文档的领域适应性检索增强生成（RAG）系统，使用AGORA语料库进行问答。我们的系统结合了检索器微调、对比学习和通过差异性优化（DPO）进行的生成器对齐。实验结果表明，尽管检索器微调改善了检索指标，但这些改进并没有 consistently 转化为更好的端到端问答性能。在多个情况下，当相关文档缺失时，强检索导致了更自信的幻觉。这些发现突显了针对政策的RAG系统面临的重要挑战：优化单个组件并不一定能保证提供更可靠、实际或有用的响应。未来的工作应探索更强的幻觉缓解策略、跨文档的上下文基础以及更好的文档状态变化处理。

# 7 限制事项

# 7.1 计算与模型规模

本研究在有限的计算资源下进行，这虽然减少了计算和能源的使用，但限制了我们能评估的模型和实验的规模。特别是，生成器基于一个具有70亿参数的指令调优模型，而不是更大规模的最先进大语言模型。尽管这一选择允许系统在相对适度的硬件上进行训练和部署，但更大的模型在政策领域问答中可能表现出不同的行为，特别是在减轻幻觉和对复杂法规文本进行推理方面。理解不同模型在性能及日常使用的物流可行性方面的优缺点，是在涉及政策文件的高风险任务中重要的考虑因素。虽然我们使用开源大语言模型以促进可重复性和降低成本障碍，但这些模型可能编码了训练数据中存在的偏见（Islam, 2026; Cuconasu et al., 2025; Blodgett et al., 2020; Brown et al., 2020）。在本研究中，我们并未对这些偏见进行明确的测量或减轻。

# 7.2 评估覆盖率

我们的评估数据集是基于对近期人工智能治理发展的政策分析和专家评论构建的。尽管这种方法提供了现实的以政策为导向的问题，但也引入了一个局限性：有些问题引用的政策尚未纳入AGORA语料库。当出现这种情况时，检索器可能会返回相关但不完整的证据，这可能导致不正确的有根据的答案。虽然这反映了不断演变的监管语料库的现实部署场景，但可能对严格依赖于现有语料库覆盖的RAG系统造成不利影响。

# 7.3 对齐的偏好数据

生成器对齐是基于相对较小的一组成对偏好注释进行的，因为在初始开发阶段领域专家的可用性有限。因此，学习到的偏好信号可能无法完全反映政策研究人员或从业者的期望。更大规模且更具多样性的专家标注偏好数据集有可能提高基于偏好的对齐在政策问答中的有效性。

# 8 伦理考虑

AGORA 数据集是开源的，完全由公共记录构建而成。我们依据其许可协议使用 AGORA，即仅限于非商业用途，并且 RAG 系统与 AGORA 团队密切合作构建，包括讨论伦理问题。尽管该系统保留了许多语言模型常见的弱点（例如，潜在的幻觉和政治偏见），但本研究的目的是原型工具，并强调此类工具在通过足够的质量检查之前，不应投入政策分析等高风险场景的生产环境。作为起点，我们强调设计与功能的透明性，开放源代码和数据，并与领域专家一起评估响应，以指导开发并制定使用指南。

# 9 致谢

我们感谢我们才华横溢的研究团队的贡献，其中包括来自普渡大学计算机科学系和政治科学系的本科生和研究生以及教职员工，他们在治理与负责任人工智能实验室工作。截至2026年3月，他们包括Mahule Roy、Ruth Sugiarto、Seunghyun Yoo、Nakshatra Tondepu、Tri Vo、Sarah Mohapatra、Selen Dogan Kosterit，以及普渡大学和乔治城大学更广泛AGORA团队的其他成员。

# References

Wasi Ahmad, Jianfeng Chi, Yuan Tian, and Kai-Wei Chang. 2020. Policyqa: A reading comprehension dataset for privacy policies. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 743749.   
Zachary Arnold, Daniel S Schiff, Kaylyn Jackson Schiff, Brian Love, Jennifer Melot, Neha Singh, Lindsay Jenkins, Ashley Lin, Konstantin Pilz, Ogadinma Enweareazu, and 1 others. 2024. Introducing the ai governance and regulatory archive (agora): An analytic infrastructure for navigating the emerging ai governance landscape. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, volume 7, pages 3948.

Arian Askari, Suzan Verberne, and Gabriella Pasi. 2022. Expert finding in legal community question answering. In European conference on information retrieval, pages 2230. Springer.

Andrew Blair-Stanek, Nils Holzenberger, and Benjamin Van Durme. 2023. Can gpt-3 perform statutory reasoning? In Proceedings of the Nineteenth International Conference on Artificial Intelligence and Law, pages 2231.

Andrew Blair-Stanek, Nils Holzenberger, and Benjamin Van Durme. 2024. Blt: Can large language models handle basic legal text? In Proceedings of the Natural Legal Language Processing Workshop 2024, pages 216232.

Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. 2020. Language (technology) is power: A critical survey of "bias" in nlp. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 54545476.

Michael Bommarito II and Daniel Martin Katz. 2022. Gpt takes the bar exam. arXiv preprint arXiv:2212.14402.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:18771901.

Nicholas Kluge Corrêa, Camila Galvão, James William Santos, Carolina Del Pino, Edson Pontes Pinto, Camila Barbosa, Diogo Massmann, Rodrigo Mambrini, Luiza Galvo, Edmund Terem, and 1 others. 2023. Worldwide ai ethics: A review of 200 guidelines and recommendations for ai governance. Patterns, 4(10).

Florin Cuconasu, Simone Filice, Guy Horowitz, Yoelle Maarek, and Fabrizio Silvestri. 2025. Do rag systems really suffer from positional bias? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2801028024.

Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E Ho. 2024. Large legal fictions: Profiling legal hallucinations in large language models. Journal of Legal Analysis, 16(1):6493.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient inetuning of quantized llms. Preprint, arXiv:2305.14314.

David Freeman Engstrom, Daniel E Ho, Catherine M Sharkey, and Mariano-Florentino Cuéllar. 2020. Government by algorithm: Artificial intelligence in federal administrative agencies. NYU School of Law, Public Law Research Paper, (20-54).

Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. 2025. Ragas: Automated evaluation of retrieval augmented generation. Preprint, arXiv:2309.15217.

Peter Henderson, Mark Krass, Lucia Zheng, Neel Guha, Christopher D Manning, Dan Jurafsky, and Daniel Ho. 2022. Pile of law: Learning responsible data filtering from the law and a 256gb open-source legal dataset. Advances in Neural Information Processing Systems, 35:2921729234.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. Preprint, arXiv:2106.09685.

Tunazzina Islam. 2026. Who gets which message? auditing demographic bias in llm-generated targeted text. arXiv preprint arXiv:2601.17172.

Zhijing Jin and Rada Mihalcea. 2022. Natural language processing for policymaking. In Handbook of computational social science for policy, pages 141162. Springer International Publishing Cham.

Minsang Kim and Seungjun Baek. 2025. Syntriever: How to train your retriever with synthetic data from llms. Preprint, arXiv:2502.03824.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, and 1 others. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459 9474.

Nestor Maslej, Loredana Fattorini, Raymond Perrault, Yolanda Gil, Vanessa Parli, Njenga Kariuki, Emily Capstick, Anka Reuel, Erik Brynjolfsson, John Etchemendy, and 1 others. 2025. Artificial intelligence index report 2025. arXiv preprint arXiv:2504.07139.

Konstantinos Mavrogiorgos, Athanasios Kiourtis, Argyro Mavrogiorgou, Georgios Manias, and Dimosthenis Kyriazis. 2024. A question answering software for assessing ai policies of oecd countries. In Proceedings of the 4th European Symposium on Software Engineering, ESSE '23, page 3136, New York, NY, USA. Association for Computing Machinery.

Gianluca Misuraca, Colin van Noordt, and Anys Boukli. 2020. The use of ai in public services: Results from a preliminary mapping across the eu. In Proceedings of the 13th international conference on theory and practice of electronic governance, pages 9099.

Theodoros Papadopoulos and Yannis Charalabidis. 2020. What do governments plan in the field of artificial intelligence? analysing national ai strategies using nlp. In Proceedings of the 13th International Conference on Theory and Practice of Electronic Governance, pages 100111.

Nicholas Pipitone and Ghita Houir Alami. 2024. Legalbench-rag: A benchmark for retrievalaugmented generation in the legal domain. arXiv preprint arXiv:2408.10343.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:5372853741.

Abhilasha Ravichander, Alan W Black, Shomir Wilson, Thomas Norton, and Norman Sadeh. 2019. Question answering for privacy policies: Combining computational and legal perspectives. arXiv preprint arXiv:1911.00841.

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. Col-BERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 37153734, Seattle, United States. Association for Computational Linguistics.

Chenhao Tang, Zhengliang Liu, Chong Ma, Zihao Wu, Yiwei Li, Wei Liu, Dajiang Zhu, Quanzheng Li, Xiang Li, Tianming Liu, and 1 others. 2023. Policygpt: Automated analysis of privacy policies with large language models. arXiv preprint arXiv:2309.10238.

ROY VAN, Fiammetta Rossetti, Karine Perset, Laura Galindo-Romero, and 1 others. AI watch-national strategies on artificial intelligence: A european perspective.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. 2020. Trl: Transformer reinforcement learning. https://github.com/huggingface/trl.

Shangrui Wang, Yuanmeng Zhang, Yiming Xiao, and Zheng Liang. 2025. Artificial intelligence policy frameworks in china, the european union and the united states: An analysis based on structure topic model. Technological Forecasting and Social Change, 212:123971.

Haoyang Wen, Jiang Guo, Yi Zhang, Jiarong Jiang, and Zhiguo Wang. 2025. On synthetic data strategies for domain-specific generative retrieval. Preprint, arXiv:2502.17957.

Jiayi Wu, Hengyi Cai, Lingyong Yan, Hao Sun, Xiang Li, Shuaiqiang Wang, Dawei Yin, and Ming Gao. 2024. Pa-rag: Rag alignment via multiperspective preference optimization. Preprint, arXiv:2412.14510.

Baobao Zhang and Allan Dafoe. 2020. US public opinion on the governance of artificial intelligence. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, pages 187193.

Haoxi Zhong, Chaojun Xiao, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, and Maosong Sun. 2020. Jecqa: a legal-domain question answering dataset. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 97019708.

# A Dataset Details

We perform data analysis to quantify the size of documents, the size of segments, the number of segments per document, and the frequencies of the tags, authorities and dates present in the dataset. We find that documents are almost all shorter than 5000 words, but some are much longer (Fig. 2a). The vast majority of segments are less than 400 words with an average length of 226 words, and more than $9 9 \%$ are less than 1000 words (Fig. 2b). There are on average 8 segments per document, and more than $9 9 \%$ of documents have less than 50 (Fig. 2c). There are 1702 unique tags in AGORA, with only 10 appearing more than 20 times (Fig. 2d). By far the most common authority is the US Congress, being the authority of more than half of all documents in AGORA (Fig. 2e). The most recent activity for all documents is no earlier than 2017, and the vast majority are from the last 3 years with a large spike at the beginning of 2025 (Fig. 2f).

# B Methodological Details

This paper proposes a RAG system constructed by fine-tuning both the retriever and the generator for AI governance and policy.

# B.1 Chunking

As the dataset has already been manually chunked by AI policy researchers into segments, we simply use the segments as chunks. The segments are already constructed to chunk each policy instrument into relatively short sections that logically divide the document. The text from the chunks is stored with various annotations and metadata attached for the retriever to also see. This allows the retriever to accurately retrieve chunks relating to questions referencing things like tags, dates, document names, and authorities. The format of a chunk can be found in Fig. 3. The "tags" field is only present if the segment has been annotated and labeled with one or more tags.

# B.2 DPO Fine-Tuning

Given a prompt $x$ , a preferred answer $y ^ { + }$ , and a rejected answer $y ^ { - }$ , DPO directly adjusts the model to increase the relative likelihood of the preferred

![](images/2.jpg)  
Figure 2: Data distribution.

![](images/3.jpg)  
Figure 3: Chunk metadata.

answer. The DPO objective is given by:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } = - \log \sigma \Big ( \beta \Big [ \log \pi _ { \theta } ( y ^ { + } \mid x ) - \log \pi _ { \theta } ( y ^ { - } \mid x ) } \\ & { ~ - \log \pi _ { \mathrm { r e f } } ( y ^ { + } \mid x ) + \log \pi _ { \mathrm { r e f } } ( y ^ { - } \mid x ) \Big ] \Big ) _ { \phantom { \theta } } } \end{array}
$$

where $\pi _ { \theta }$ is the fine-tuned policy model, $\pi _ { \mathrm { r e f } }$ is a frozen reference copy of the base model, $\beta$ controls how far the model is allowed to deviate from the reference, and $\sigma$ is the sigmoid function. In Fig. 6, after question/answer generation and collection of preferences, the base LLM (Mistral-7b-Instructv0.3) is trained according to the objective above.

# B.2.1 Training Data Generation

Questions for the training data are generated document-wise from the AGORA dataset, where we prompt an LLM to create a few questions based on that document, a given category, and an example format for that theme. Because the retrieval capabilities of the RAG system are not tested, we choose to provide the model with the reference document when it was generating the answer pairs for

the questions.

To generate the training data for DPO finetuning, two distinct model configurations are used to produce responses of differing styles and quality. Both configurations are based on the same underlying model, but vary in their prompting and decoding strategies to promote diversity in outputs. The first configuration is designed to produce detailed, well-reasoned, and comprehensive responses grounded in the provided context. Its prompt positions the model as an "expert policy analyst" expected to deliver structured explanations and incorporate multiple perspectives. In contrast, the second configuration emphasizes brevity and clarity, encouraging concise and direct answers with minimal elaboration. This setup aims to capture more succinct, readable responses that might sacrifice depth for precision.

# B.2.2 Fine-Tuning Process

After we store the pairwise preferences for each model response, we use those to align our LLM with DPO. Each preference is stored as a triplet (prompt, chosen, rejected), where the prompt used is the one that encouraged detailed answers, including the document context. For computational efficiency, we load the Mistral-7b-Instruct model in an 8-bit quantization, which helps with GPU memory limitations during training. To run DPO efficiently on a single GPU, we use parameter-efficient fine-tuning (PEFT) with LoRA (Hu et al., 2021) adapters, meaning we only have to update a small number of low-rank matrices rather than the full model parameters. Furthermore, we use a gradient accumulation of 8 steps with a batch size of 2, simulating a batch size of 16. We find that 1 epoch is a sufficient training time, as training for longer over a smaller preference dataset, as we have in our case, can lead to overfitting of annotation noise rather than improving alignment.

![](images/4.jpg)  
Figure 4: Retriever training pipeline.

# B.3 Retriever Fine-tuning

As visualized in Fig. 4, the retriever fine-tuning process is a pipeline of multiple steps to optimize a retriever for the AGORA dataset in an automatic process not requiring manual intervention.

# B.3.1 Synthetic Query Generation

For both the issue of evaluating a retriever as well as fine-tuning one, a set of queries with known relevant documents is necessary. As manually constructing a large set of these labeled queries requires significant time and effort, synthetic query generation is an efficient and effective method to create such a dataset (Wen et al., 2025; Kim and Baek, 2025). We construct a synthetic query generation pipeline specifically for retriever training and evaluation that creates queries with some relevant coverage over the range of possible user queries, and manually labeled a set of these queries with relevant and irrelevant documents. The question generation consists of a prompt generation system that creates thousands of prompts for an LLM to create queries for the retriever. Considering the research-focused use case of our system, we create prompts focused on analysis-related topics like trends and comparisons. With the AGORA dataset already being annotated at the document and chunk level with things like tags, authorities, and dates, combinations of those annotations are inserted into the prompts to create full prompts for the LLM that cover many possible topics within the scope of AI governance and regulation. These filled prompts are then passed to the LLM with instructions to create a question based on the prompt.

# B.3.2 Positive/Negative Example Labeling

To fine-tune ColBERTv2, we need to label the queries with positive and negative examples from the chunks in the dataset. We discard queries if they are not well-formed or just generally not relevant. With questions that we have not discarded, we use ColBERTv2 to retrieve the top-20 chunks for each question, and manually label each chunk as relevant or irrelevant. This obviously gives no guarantee of finding the best positive and negative examples, but by manually labeling, we ensure that the training moves the retriever in the right direction. After this labeling process, (query, positive example, negative example) triples are created by creating a triple from each possible pair of positive and negative examples from the labeled sets of each query. For this reason, despite only 127 queries being labeled, 8339 training triples are created. Furthermore, the Ragatouille library makes available a hard negative mining feature that finds likely useful negative examples for any given query. This gave us three options for providing negative examples to the fine-tuning procedure. First, just using our labeled negatives and not using the mining procedure at all, second, only using the mined negatives and disregarding our labeled negatives, and third, using both the mined and labeled negatives. We fine-tune a retriever using each of these three methods. We refer to these fine-tuned retriever variants as "Labeled negatives", "Mined negatives", and "Mixed negatives".

# B.3.3 Fine-tuning Procedure

ColBERTv2 based retriever is fine-tuned using our synthetically generated and manually labeled query dataset using the Ragatouille libraries' RAGTrainer feature. The RAGTrainer performs a contrastive learning procedure to update the parameters of the ColBERTv2 retriever to make the embedding of the query closer to the embedding for the positive example and further from the negative example. It does this by optimizing a contrastive InfoNCE objective (2) that increases the ColBERT similarity $S ( q , p )$ (1) between a query and its positive passage while decreasing similarity to negatives.

$$
\begin{array} { r l r } {  { S ( q , p ) = \sum _ { t } \operatorname* { m a x } _ { s } \sin ( u _ { t } , v _ { s } ) , } } & { { } } & { ( \mathrm { 2 } } \\ { \qquad } & { { } } & { \mathcal { L } ( q , p ^ { + } , p ^ { - } ) = - \log \frac { e ^ { S ( q , p ^ { + } ) / \tau } } { e ^ { S ( q , p ^ { + } ) / \tau } + e ^ { S ( q , p ^ { - } ) / \tau } } . } \end{array}
$$

In these equations, $q$ denotes a query and $p$ a passage. The query and passage are encoded into token-level embeddings, where $u _ { t }$ is the embedding of the $t$ -th query token and $v _ { s }$ is the embedding of the $s$ -th passage token. The similarity function $\sin ( u _ { t } , v _ { s } )$ measures the similarity between token embeddings. For training, the $\mathcal { L } ( g , p ^ { + } , p ^ { - } )$ contrasts a relevant (positive) passage $p ^ { + }$ against an irrelevant (negative) passage $p ^ { - }$ , using a temperature parameter $\tau$ to scale the logits.

# B.4 RAG pipeline

The system itself is a fairly standard RAG pipeline using the fine-tuned retriever and generator. The AGORA dataset is chunked simply by using the document segments as chunks. The segments are manually created by researchers when documents are added to the dataset to split the documents into semantically relevant chunks of a reasonable length, so simply using the segments as our chunks is the most logical decision. The chunks are encoded into an index using our fine-tuned retriever using the

Ragatouille library's RAGPretrainedModel.index function. When running, the system receives user queries, encodes them using the fine-tuned retriever, and then using RAGPretrainedModel.search finds the top-20 chunks from the index. These chunks are passed to the fine-tuned generator along with the user query, and the generator responds to the user query.

# C Experimental Details

# C.1 Evaluation Question Creation

To evaluate our system, we need a set of relevant questions labeled with factually accurate answers. Based on the size of our document corpus, we determined that a set of 300 questions and answers is a suitable size. We create the questions and answers from factual analysis of AI policy found in the AI Policy Corner2 blog of the Montreal AI Ethics Institute. These articles feature analysis discussing the content and implications of recent AI policies from around the world, and comparing the approaches taken through different policies. Our process for creating an evaluation dataset for our system involved taking factual statements from these articles and creating questions about AI policy that those factual statements accurately answer. This is done both manually and with assistance from LLMs. In this process, we are assisted by contributions from policy experts involved in the writing of the articles.

# C.2 Generator

# C.2.1 Generator Optimization

To generate the set of questions for the DPO training data, the gemini-2.0-flash and gemini-2.5-flashlite models are used. For the documents, to avoid overloading the context length, only documents with word counts between 300 and 1200 words are chosen. The six categories that each question fell under are Summarization/Explanation, Implication, Stakeholder Interpretation, Definitions in Context, Compliance, and Evaluation, with each category having 299, 305, 349, 342, 368, and 337 questions, respectively, making a total of 2000 questions. In the prompt provided to the model to make the questions, the same general structure is preserved across categories, and during generation, along with specifying the category, an example question is added

# Q3: How does this act define 'automated employment decision tool' and what types of systems does this definition encompass?

for the model to reference. Example questions provided to the model for each category are shown in Fig. 7

![](images/5.jpg)  
Figure 5: DPO response collection GUI.

![](images/6.jpg)  
Figure 6: Generator DPO fine-tuning pipeline.

When actually generating the answer pairs for the questions, Mistral-7B-Instruct- $\cdot \mathrm { v } 0 . 3$ is used, which is the same model that was fine-tuned. The two different model configurations consisted of a different system prompt, as well as different choices for the temperature, $t o p _ { \mathrm { \mathcal { P } } }$ , and top $\mathbf { \Omega } _ { k }$ hyperparameters. The first configuration used temper-$a t u r e = 0 . 2$ , top_ $p = 0 . 9 5$ , and $t o p \_ k = 4 0$ , paired with a prompt that emphasized detailed information grounded in the provided context. With the lower temperature and the higher top_k, we intended this model configuration to generate in-depth explanations while still considering multiple points of view. The second configuration used temperature $= 0 . 9$ $t o p \_ p = 0 . 6$ ,and $t o p \_ k = 2 0$ , along with a prompt that instructed the model to create brief responses to have more readable responses. The prompts used both for adding more detail and being more concise are shown in Fig. 7

After the generation of these answers, human preferences are collected through a streamlit interface as seen in Fig. 5, where the user can see the document context along with the questions and answers.

# C.2.2 DPO Training Loop

We load the base model, Milstral-7b-Instruct-v2, in 8-bit quantized mode through the bitsandbytes Python library. Because fully quantized models cannot be fine-tuned directly, we added LoRA adapters with the PEFT library. This setup matches the QLoRA (Dettmers et al., 2023) configuration, where the 8-bit backbone remains frozen and only the LoRA layers (in FP16) are trainable. This training paradigm achieves a lot of memory savings, allowing us to fine-tune on a single A100 40GB GPU.

We use HuggingFace TRL's (von Werra et al., 2020) DPOTrainer, which automatically created a frozen reference model, handled the DPO objective along with the training loop, batching, and logging. As for hyperparameters, using a batch size of 2 with a gradient accumulation value of 8 led to an effective batch size of 16; the learning rate was $5 \times 1 0 ^ { - 6 }$ ; we trained for 1 epoch, the $\beta$ value was 0.1; and the optimizer was paged_adamw_32bit, a memory-efficient optimizer for quantized models.

# C.2.3 Generator Evaluation

To evaluate the generator without relying on ground-truth answers, we rely on one key metric: faithfulness. Faithfulness evaluates whether the generated response is supported by the retrieved context by decomposing the answer into individual claims and verifying that each is grounded in the provided evidence. With this metric, we essentially evaluate the generator's ability not to hallucinate and be truthful. The evaluation was conducted using RAGAS, an evaluation framework for RAG systems. The test set of 208 questions was generated using the almost exact methodology as the training set for DPO; the only difference was using longer document lengths ( $1 2 0 0 +$ words) so that there would be no overlap between the documents used for the training data (300-1200 words). For question categories, the same categories as the training data are used, and questions of each type are randomly distributed. After responses from the candidate models are collected, it is passed into the RAGAS faithfulness metric, where an LLM is used to extract all claims in the answer and verify what percent of those claims can be inferred from the given context. The LLM used to evaluate the responses is also Mistral-7B-Instruct-v0.3.

# C.3 Retriever

# C.3.1 Synthetic Query Generation

As discussed in B.3, we need to synthetically generate queries to use to both train and evaluate the retriever. To do this, we use gemma3:27b to generate questions based on a set of prompts. The prompts contain places for tags, authorities, and dates to be inserted. The prompts used can be found in Fig. 8. Any text inside a set of curly brackets is a fillable part of the prompt, and any text inside angle brackets may or may not be included in a given prompt. Tags, authorities, and dates are randomly generated to fill the prompts as needed. The questions generated in this synthetic query generation procedure are split into a training group and a test group.

# C.3.2 Evaluation Question Labeling

To create a set of queries labeled with relevant chunks to use to evaluate our fine-tuned retrievers, we use a similar manual labeling procedure as discussed in B.3.2, but instead of having ColBERTv2 retrieve the top 20 chunks per query, it retrieves the top-50. We create a set of 50 labeled test questions in this process. As with the training query labeling, we are also able to discard irrelevant or poorly formed questions. Each of these 50 chunks per query is manually marked as relevant or irrelevant. Furthermore, as we are basing the relevant chunk set on base ColBERTv2, we note that ColBERTv2 will have an inherent advantage over the fine-tuned retrievers in our evaluation process.

# C.3.3 Retriever Evaluation

As discussed in B.3.2, we fine-tune ColBERTv2 in 3 different ways using 3 different methods to find negative examples for each query. We evaluate these retriever variants within our RAG system using Mean Reciprocal Rank (MRR), Recall $@ \mathbf { k }$ , and mean average precision at k $( \mathbf { M A P } @ \mathbf { k } )$ (for k $= 5 , 1 0 { , } 2 0 )$ on the set of 50 synthetically generated and manually labeled test questions.

# C.4 System Error Analysis

We analyze system error analysis on the evaluation set in several cases.

# C.4.1 Documents Not in AGORA

In this case, the question is "Which FY 2026 NDAA section creates (or allows creation of) AI research institutes?" The FY 2026 NDAA document is not in the AGORA dataset currently, so it should not be retrievable. However, our retriever still retrieves the NDAA from previous years, so our system mentions the establishment of AI research institutes in previous years. In some cases, the generator erroneously classifies the results from previous years as 2026, which the DPO fine-tuning helps with, as this error only occurs with the base generators.

# C.4.2 Incorrect Country Retrieval/Generation

In a few cases, the question is related to one country, yet the retriever extracted context from another country, and the generator incorrectly attributed this information to the original country. For example, for the query "What role is the public sector expected to play in accelerating AI adoption in South Korea (e.g., procurement or deployment incentives)?", the retriever pulls documents from countries such as the United States, Singapore, China, and Australia. It should be noted that documents regarding the public sector in South Korea were absent in the AGORA dataset used for analysis. Though the generator is given irrelevant documents, it should still have identified that this information is not pertinent, but it instead takes the context it is given from the unrelated countries, and uses that to generate the answer to the question as if it were discussing South Korea. In other similar cases, the generator mentions that it is not given context relevant to the situation, and then discusses related documents that are retrieved in a contextually-appropriate way, making sure to mention the true source country.

# C.5 Retriever Error Analysis

We analyze an erroneous retrieval from each of the 3 trained retrievers.

# C.5.1 Mined negatives retriever

The question here is "What key themes or areas of focus have emerged in AI-related policies enacted by the Commonwealth of Virginia since January 1, 2023?" The sixth chunk retrieved by the mined negatives retriever is segment_1459_2. This is a segment from the West Virginia HB 5690 (Artificial Intelligence Task Force). This was proposed in 2024, but obviously is from West Virginia, not Virginia. This is an understandable mistake due to the similarity of the state names. This type of error with misidentifying authorities is a common error that the fine-tuning helped with, but is a serious concern likely to be pertinent in policy contexts.

# C.5.2 Labeled negatives retriever

The question here is "How have pilot programs and testbeds been utilized by organizations \*other than\* major governmental bodies to explore and shape AI policy over time?" The first 3 retrieved chunks all come from document_307, which is the 2023 US "Executive Order on the Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence (EO 14110)". This is not relevant as it obviously is from a major governmental body. Once again, this is an issue of authority; in this case, the difficulty likely comes from the wording of the question. The authority needs to be something "\*other than\* major governmental bodies," which excludes rather than explicitly naming a particular authority. This requires a more nuanced interpretation of the query that is more difficult to fine-tune into the retriever.

# C.5.3 Mixed negatives retriever

The question here is "Considering the provisions outlined in both the 'Duplicative Grant Consolidation Act' and Arizona's Senate Bill 1359, how might the use of artificial intelligence be leveraged both to prevent financial misuse \*and\* to address potential misinformation campaigns, particularly concerning public figures?" The first re-

# Generator optimization query generation question examples

Summarization/Explanation: What problem does {policy_name} aim to address?" /"Summarize the main objectives of {policy_name}.

Implication: What are the expected impacts of {policy_name} on {group}?

Definitions In Context: What does {term} mean under {policy_name}, and why is it significant?

Compliance: What mechanisms ensure compliance under {policy_name}?

valuation: How effective is {policy_name}likely to be in addressing {issue}?

Generator optimization long query generation prompt

You are an expert policy analyst tasked with answering questions about Al policy and regulations. Provide direct, factual information grounded in the context. Cite relevant sources or document IDs where applicable. If the context does not contain enough information, state that explicitly instead of speculating.

Context: {context}   
Question: {question}   
Provide: a comprehensive answer with a direct answer to the question, and citations to relevant sources where necessary.   
Answer:

Generator optimization short query generation prompt

You are a concise and formal assistant providing brief, factual answers about Al policy. Answer directly based only on the given context. If the context is insufficient, say so clearly without guessing.

Context: {context}

Question: {question}

Answer in 3-5 sentences citing document IDs where applicable

Answer:

# Retriever finetuning synthetic query generation system prompt

You are an Al policy expert creating questions to use to train and evaluate a retriever in a RAG pipeline. You will be given a description of the type of question to ask, and possibly the text and title of one or more Al policy documents that the RAG system should retrieve to answer the question. You may be provided with tags, that are used by researchers to categorize policy documents, and authorities, which are governments or organizations that create policy. Be creative. The questions you generate should be worded in different ways. Respond each time with one single question.

Retriever finetuning synthetic query generation annotation prompts

Tag tatus: Ask a question about the status of Al policy relatingto tag: {ag} created by the authority of: {authority}.

Trends: Ask one single question about Al policy trends <on policy created by authority: {authority}> <relating to tag: {tag}> <since {year}.>

Authority comparison: Ask a question about the differences in Al policy <relating to tag: {tag}> between policy created by authorities {auth_1} and {auth_2}

Retriever finetuning synthetic query generation document prompts trieved chunk is segment_1535_4. This is from "Utah Senate Bill 131 Information Technology Act Amendments (2024)." This obviously is not one of the documents listed in the question, and it is not entirely clear why it was the top chunk retrieved. It is possible that the query listing multiple documents and question length made accurate retrieval difficult, or that irrelevant text similarity confounded the retriever.

![](images/7.jpg)  
Figure 8: Retriever fine-tuning prompts.

# D Expert Analysis of System

We provide example questions and generated answers from our system to policy researchers who are experts on AI policy. One researcher with expertise specifically in Turkey and the EU provided us with reviews of the answers to questions relating to those two documents. The following 3 questions were provided:

•Question 1: Discuss the similarities and differences between Turkey's National AI strategy and the EU AI Act in terms of risktiering?

• Question 2: Compare Turkey's National AI Strategy proposal's comprehensiveness to other national-level AI Acts.

• Question 3: How does Turkey's National AI Strategy fall short of international standards? On which aspects can it be improved?

The following is the written feedback that we received from the policy expert, which was echoed by other policy experts as part of a workshop:

Assessment of Question 1 Answer: "Turkey's National AI strategy does not explicitly mention a risk-tiering approach, which the chatbot writes in the middle of its response. However, it starts the answer by saying "the EU AI Act and Turkey's National AI strategy both employ a risk-tiering approach". So I think the answer contradicts itself a little bit. Moreover, I wish the chatbot would capture the exact mechanism of the EU AI Act risk-tiering approach (EU AI Act categorizes AI systems into 4 groups based on risk: unacceptable risk, high risk, limited risk and minimal risk. While unacceptable risk AI systems are prohibited, high risk AI systems are subject to conformity assessment, etc.) I think this kind of answer should have been more accurate to describe the EU AI Act risktiering."

Assessment of Question 2 Answer: "I overall liked this answer. The chatbot was able to capture the main points of the National AI strategy, and it even provided evidence from specific segments. It also rightfully pointed out that EU focuses on transparency, accountability, human oversight, and prohibiting certain uses of AI; which is lacking in Turkey. But I wished the answer was specifically pointing out that Turkey's National AI strategy does not mention ANY HARMS that could be caused during the development and use of AI systems. I think this was the main missing point in the answer.

The chatbot also pointed out that "specific actions and priorities may vary depending on the country's unique needs and circumstances," which I found interesting, because it seems like the chatbot does not take a normative stance in terms of which actions/priorities should be necessary in an AI regulatory framework."

Assessment of Question 3 Answer: "I liked this answer the most and found it quite comprehensive. It was not only summarizing which aspects Turkey's document did well, but also recommending how to enhance it to meet international standards better after each specific point. (For instance, it mentioned how the strategy includes various actions to train AI experts and increase employment in the field. Then, it was giving suggestions to improve this, such as incorporating more emphasis on lifelong learning and continuous professional development to ensure the workforce remains up-to-date with the latest AI technologies and trends.)

I liked most of its suggestions as I could see it was drawing on the strengths of other international standards, but it was not really naming WHICH international standards. So, one way to improve this answer might be actually writing something like "The strategy can be improved by ensuring that AI systems are accessible to all, regardless of socio-economic background. Some international standards highlight the accessibility aspect and how AI systems should be made accessible to vulnerable groups - for example, the South Korean AI Act." In short, actually giving examples of other international standards might be quite helpful in answering a question like this, since the prompt is clearly asking how the Turkish strategy falls short of international standards."
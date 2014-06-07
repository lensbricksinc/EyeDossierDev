//# include "svm_common.h"
//
//int svm_classify (char *line, MODEL* model) {
//	
//	DOC *doc;   /* test example */
//	WORD *words;
//	long max_docs,max_words_doc,lld;
//	long totdoc=0,queryid,slackid;
//	long correct=0,incorrect=0,no_accuracy=0;
//	long res_a=0,res_b=0,res_c=0,res_d=0,wnum;
//	long j;
//	double t1,runtime=0;
//	double dist,doc_label,costfactor;
//	char *comment;
//
//	words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc+10));
//
//
//	if(model->kernel_parm.kernel_type == 0) { /* linear kernel */
//	/* compute weight vector */
//	add_weight_vector_to_linear_model(model);
//	}
//
//	if(line[0] == '#') return 0;  /* line contains comments */
//	parse_document(line,words,&doc_label,&queryid,&slackid,&costfactor,&wnum,
//			max_words_doc,&comment);
//	totdoc++;
//	if(model->kernel_parm.kernel_type == 0) {   /* linear kernel */
//		for(j=0;(words[j]).wnum != 0;j++) {  /* Check if feature numbers   */
//	if((words[j]).wnum>model->totwords) /* are not larger than in     */
//		(words[j]).wnum=0;               /* model. Remove feature if   */
//		}                                        /* necessary.                 */
//		doc = create_example(-1,0,0,0.0,create_svector(words,comment,1.0));
//		t1=get_runtime();
//		dist=classify_example_linear(model,doc);
//		free_example(doc,1);
//	}
//	else {                             /* non-linear kernel */
//		doc = create_example(-1,0,0,0.0,create_svector(words,comment,1.0));
//		t1=get_runtime();
//		dist=classify_example(model,doc);
//		runtime+=(get_runtime()-t1);
//		free_example(doc,1);
//	}
//	if(dist>0) {
//      
//		if(doc_label>0) correct++; else incorrect++;
//		if(doc_label>0) res_a++; else res_b++;
//	}
//	else {
//      
//		if(doc_label<0) correct++; else incorrect++;
//		if(doc_label>0) res_c++; else res_d++;
//	}
//    
//	if((int)(0.01+(doc_label*doc_label)) != 1) 
//		{ no_accuracy=1; } /* test data is not binary labeled */
//	if(verbosity>=2) {
//		if(totdoc % 100 == 0) {
//	printf("%ld..",totdoc); fflush(stdout);
//		}
//	}
//
//	free(line);
//	free(words);
//	free_model(model,1);
//
//	if((!no_accuracy)) {
//	printf("Accuracy on test set: %.2f%% (%ld correct, %ld incorrect, %ld total)\n",(float)(correct)*100.0/totdoc,correct,incorrect,totdoc);
//	printf("Precision/recall on test set: %.2f%%/%.2f%%\n",(float)(res_a)*100.0/(res_a+res_b),(float)(res_a)*100.0/(res_a+res_c));
//	}
//
//	return(0);
//}
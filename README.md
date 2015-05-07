## Edit and Preview the Website

All documentations are written in Markdown format and located at _posts/.

### Preview locally

If you have installed Jekyll, then you can preview the website by

    jekyll serve --config _config.yml,_config-prod.yml

### Preview on your own Github site

If you do not have Jekyll installed on your own computer, then your can follow
these steps to preview the website on you own Github site:

 * fork SINGA to your own Github account from [https://github.com/apache/incubator-singa](https://github.com/apache/incubator-singa).
 * clone SINGA from your Github to your local computer
 * edit the content (e.g., add documentations)
 * update the BASE_URL in _config.yml to "http://username.github.io/incubator-singa"
 * commit and push to your own Github repo
 * goto http://username.github.io/incubator-singa/

## Submit to the Incubator-Singa repo

After finishing updating the website, commiters can submit their commits as
follows.
(*Make sure that you have changed back the BASE_URL to
"http://apache.github.io/incubator-singa" if you have previewed.
it on your own Github site.*)

    # done only once
    git add upstream git://git.apache.org/incubator-singa.git

    # done for every submit
    git push upstream gh-pages

## License

* We used Jekyll-Boostrap [MIT](http://opensource.org/licenses/MIT) to generate this website.
* The source code except that from Jekyll-Boostrap is release under
[Apache License 2](http://www.apache.org/licenses/LICENSE-2.0.html).

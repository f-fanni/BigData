sed -E 's/<p>|<q>|<\/p>|<\/q>|<a [^>]*>|<\/a>/ /g; s/<article [^>]*>/&<content>/g ; s/<\/article>/<\/content>&/g' ../dataset/articles-training-bypublisher-20181122.xml > ../dataset/wellformed/articles-training-bypublisher.xml

sed -E 's/<p>|<q>|<\/p>|<\/q>|<a [^>]*>|<\/a>/ /g; s/<article [^>]*>/&<content>/g ; s/<\/article>/<\/content>&/g' ../dataset/articles-training-byarticle-20181122.xml > ../dataset/wellformed/articles-training-byarticle.xml

sed -E 's/<p>|<q>|<\/p>|<\/q>|<a [^>]*>|<\/a>/ /g; s/<article [^>]*>/&<content>/g ; s/<\/article>/<\/content>&/g' ../dataset/articles-validation-bypublisher-20181122.xml > ../dataset/wellformed/articles-validation-bypublisher.xml

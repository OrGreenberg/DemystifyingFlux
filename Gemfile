# frozen_string_literal: true

source "https://rubygems.org"

git_source(:github) do |repo_name|
  repo_name = "#{repo_name}/#{repo_name}" unless repo_name.include?("/")
  "https://github.com/#{repo_name}.git"
end

# Jekyll
gem "jekyll"

# Theme (replace minima with your theme if different)
gem "minima"

# For Jekyll plugins (like jekyll-feed, jekyll-sitemap, etc.)
group :jekyll_plugins do
  # Add any plugins here. Example:
  # gem "jekyll-feed"
  # gem "jekyll-sitemap"
end

<script setup lang="ts">
interface Reference {
  id: string
  authors: string
  title: string
  venue: string
  year: string | number
  url?: string
}

const props = defineProps<{
  references: Reference[]
}>()
</script>

<template>
  <div class="reference-block">
    <ol class="reference-list">
      <li v-for="ref in references" :key="ref.id" class="reference-item">
        <span class="ref-id">[{{ ref.id }}]</span>
        <span class="ref-content">
          <span class="ref-authors">{{ ref.authors }}</span>.
          <span v-if="ref.url">
            <a :href="ref.url" target="_blank" rel="noopener" class="ref-title-link">{{ ref.title }}</a>.
          </span>
          <span v-else class="ref-title">{{ ref.title }}.</span>
          <span class="ref-venue">{{ ref.venue }}</span>,
          <span class="ref-year">{{ ref.year }}</span>.
        </span>
      </li>
    </ol>
  </div>
</template>

<style scoped>
.reference-block {
  background: var(--vp-c-bg-alt, #161b22);
  border: 1px solid var(--vp-c-border, #30363d);
  border-radius: 12px;
  padding: 20px 24px;
  margin: 24px 0;
}

.reference-list {
  list-style: none;
  padding: 0;
  margin: 0;
  counter-reset: ref-counter;
}

.reference-item {
  display: flex;
  gap: 10px;
  padding: 10px 0;
  border-bottom: 1px solid var(--vp-c-border, #30363d);
  font-size: 0.88rem;
  line-height: 1.5;
  color: var(--vp-c-text-2, #8b949e);
}

.reference-item:last-child {
  border-bottom: none;
}

.ref-id {
  font-weight: 700;
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
  flex-shrink: 0;
  min-width: 32px;
}

.ref-content {
  flex: 1;
}

.ref-authors {
  font-weight: 600;
  color: var(--vp-c-text-1, #e6edf3);
}

.ref-title {
  font-style: italic;
  color: var(--vp-c-text-1, #e6edf3);
}

.ref-title-link {
  font-style: italic;
  color: var(--vp-c-brand-1);
  text-decoration: none;
  transition: color 0.2s ease;
}

.ref-title-link:hover {
  color: var(--vp-c-brand-3);
  text-decoration: underline;
}

.ref-venue {
  font-weight: 500;
}

.ref-year {
  font-weight: 700;
  color: var(--vp-c-text-1, #e6edf3);
}
</style>

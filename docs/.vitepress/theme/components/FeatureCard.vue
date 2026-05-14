<script setup lang="ts">
const props = defineProps<{
  icon?: string
  title: string
  description: string
  tags?: string[]
  link?: string
}>()

const tagsList = props.tags ?? []
</script>

<template>
  <component
    :is="link ? 'a' : 'div'"
    :href="link"
    class="feature-card-component"
    :class="{ 'is-link': !!link }"
  >
    <div class="card-header">
      <span v-if="icon" class="card-icon">{{ icon }}</span>
      <h3 class="card-title">{{ title }}</h3>
    </div>
    <p class="card-description">{{ description }}</p>
    <div v-if="tagsList.length" class="card-tags">
      <span v-for="(tag, i) in tagsList" :key="i" class="tag-pill">{{ tag }}</span>
    </div>
  </component>
</template>

<style scoped>
.feature-card-component {
  display: block;
  background: var(--vp-c-bg-alt, #161b22);
  border: 1px solid var(--vp-c-border, #30363d);
  border-radius: 12px;
  padding: 24px;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  text-decoration: none;
  color: inherit;
}

.feature-card-component.is-link {
  cursor: pointer;
}

.feature-card-component:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3), 0 0 0 1px var(--vp-c-brand-soft);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.card-icon {
  font-size: 1.75rem;
  line-height: 1;
}

.card-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--vp-c-text-1, #e6edf3);
  margin: 0;
}

.card-description {
  font-size: 0.9rem;
  color: var(--vp-c-text-2, #8b949e);
  line-height: 1.5;
  margin: 0 0 14px 0;
}

.card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tag-pill {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 999px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
</style>

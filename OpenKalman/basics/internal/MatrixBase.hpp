/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition of MatrixBase.
 */

#ifndef OPENKALMAN_MATRIXBASE_HPP
#define OPENKALMAN_MATRIXBASE_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Ultimate base of typed matrices and covariance matrices.
   * \tparam Derived The fully derived matrix type.
   * \tparam NestedMatrix The nested native matrix, which can be const or an lvalue reference, or both, or neither.
   */
#ifdef __cpp_concepts
  template<indexible Derived, indexible NestedMatrix> requires (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Derived, typename NestedMatrix>
#endif
  struct MatrixBase : internal::library_base_t<Derived, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif


    /**
     * \internal
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr MatrixBase() noexcept requires self_contained<NestedMatrix> and std::default_initializable<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<self_contained<T> and
      std::is_default_constructible<NestedMatrix>::value, int> = 0>
    constexpr MatrixBase() noexcept
#endif
      : m_arg {} {}


    /**
     * \internal
     * \brief Construct from a nestable type.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<typename Arg> requires (not std::is_base_of_v<MatrixBase, std::decay_t<Arg>>) and
    std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<not std::is_base_of_v<MatrixBase, std::decay_t<Arg>> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    constexpr explicit MatrixBase(Arg&& arg) noexcept : m_arg {std::forward<Arg>(arg)} {}


    /**
     * \brief Get the nested object.
     */
    constexpr auto& nested_object() & noexcept { return m_arg; }


    /// \overload
    constexpr const auto& nested_object() const & noexcept { return m_arg; }


    /// \overload
    constexpr auto&& nested_object() && noexcept { return std::move(*this).m_arg; }


    /// \overload
    constexpr const auto&& nested_object() const && noexcept { return std::move(*this).m_arg; }


    /**
     * \brief Access a component at a set of indices.
     * \return If <code>writable_by_component<Derived, Indices></code>, the component can be directly assigned.
     */
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      (static_range_size_v<Indices> == dynamic_size or index_count_v<Derived> == dynamic_size or
        static_range_size_v<Indices> >= index_count_v<Derived>)
#else
    template<typename Indices, std::enable_if_t<index_value<decltype(*std::declval<Indices>().begin())> and
      (static_range_size<Indices>::value == dynamic_size or index_count<Derived>::value == dynamic_size or
        static_range_size<Indices>::value >= index_count<Derived>::value), int> = 0>
#endif
    constexpr auto operator()(const Indices& indices) &
    {
      if constexpr (writable_by_component<Derived, Indices>) return ElementAccessor(static_cast<Derived&>(*this), indices);
      else return get_component(static_cast<Derived&>(*this), indices);
    }


    /// \overload
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      (static_range_size_v<Indices> == dynamic_size or index_count_v<Derived> == dynamic_size or
        static_range_size_v<Indices> >= index_count_v<Derived>)
#else
    template<typename Indices, std::enable_if_t<index_value<decltype(*std::declval<Indices>().begin())> and
      (static_range_size<Indices>::value == dynamic_size or index_count<Derived>::value == dynamic_size or
        static_range_size<Indices>::value >= index_count<Derived>::value), int> = 0>
#endif
    constexpr auto operator()(const Indices& indices) &&
    {
      if constexpr (writable_by_component<Derived&&>) return ElementAccessor(static_cast<Derived&&>(*this), indices);
      else return get_component(static_cast<Derived&&>(*this), indices);
    }


    /// \overload
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      (static_range_size_v<Indices> == dynamic_size or index_count_v<Derived> == dynamic_size or
        static_range_size_v<Indices> >= index_count_v<Derived>)
#else
    template<typename Indices, std::enable_if_t<index_value<decltype(*std::declval<Indices>().begin())> and
      (static_range_size<Indices>::value == dynamic_size or index_count<Derived>::value == dynamic_size or
        static_range_size<Indices>::value >= index_count<Derived>::value), int> = 0>
#endif
    constexpr auto operator()(const Indices& indices) const &
    {
      return get_component(static_cast<const Derived&>(*this), indices);
    }


    /// \overload
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
    template<std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>> and
      (static_range_size_v<Indices> == dynamic_size or index_count_v<Derived> == dynamic_size or
        static_range_size_v<Indices> >= index_count_v<Derived>)
#else
    template<typename Indices, std::enable_if_t<index_value<decltype(*std::declval<Indices>().begin())> and
      (static_range_size<Indices>::value == dynamic_size or index_count<Derived>::value == dynamic_size or
        static_range_size<Indices>::value >= index_count<Derived>::value), int> = 0>
#endif
    constexpr auto operator()(const Indices& indices) const &&
    {
      return get_component(static_cast<const Derived&&>(*this), indices);
    }


    /**
     * \brief Access a component at a set of indices.
     * \return If <code>writable_by_component<Derived></code>, the component can be directly assigned.
     */
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      (index_count_v<Derived> == dynamic_size or sizeof...(I) >= index_count_v<Derived>) and
      internal::static_indices_within_bounds<Derived, I...>::value
#else
    template<typename...I, std::enable_if_t<(index_value<I> and ...) and
      (index_count<Derived>::value == dynamic_size or sizeof...(I) >= index_count<Derived>::value) and
    internal::static_indices_within_bounds<Derived, I...>::value, int> = 0>
#endif
    constexpr auto operator()(I&&...i) &
    {
      const auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      if constexpr (writable_by_component<Derived, std::array<std::size_t, sizeof...(I)>>)
        return ElementAccessor(static_cast<Derived&>(*this), indices);
      else return get_component(static_cast<Derived&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      (index_count_v<Derived> == dynamic_size or sizeof...(I) >= index_count_v<Derived>) and
      internal::static_indices_within_bounds<Derived, I...>::value
#else
    template<typename...I, std::enable_if_t<(index_value<I> and ...) and
      (index_count<Derived>::value == dynamic_size or sizeof...(I) >= index_count<Derived>::value) and
      internal::static_indices_within_bounds<Derived, I...>::value, int> = 0>
#endif
    constexpr auto operator()(I&&...i) &&
    {
      const auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      if constexpr (writable_by_component<Derived, std::array<std::size_t, sizeof...(I)>>)
        return ElementAccessor(static_cast<Derived&&>(*this), indices);
      else return get_component(static_cast<Derived&&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      (index_count_v<Derived> == dynamic_size or sizeof...(I) >= index_count_v<Derived>) and
      internal::static_indices_within_bounds<Derived, I...>::value
#else
    template<typename...I, std::enable_if_t<(index_value<I> and ...) and
      (index_count<Derived>::value == dynamic_size or sizeof...(I) >= index_count<Derived>::value) and
      internal::static_indices_within_bounds<Derived, I...>::value, int> = 0>
#endif
    constexpr auto operator()(I&&...i) const &
    {
      const auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      return get_component(static_cast<const Derived&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value...I> requires
      (index_count_v<Derived> == dynamic_size or sizeof...(I) >= index_count_v<Derived>) and
      internal::static_indices_within_bounds<Derived, I...>::value
#else
    template<typename...I, std::enable_if_t<(index_value<I> and ...) and
      (index_count<Derived>::value == dynamic_size or sizeof...(I) >= index_count<Derived>::value) and
      internal::static_indices_within_bounds<Derived, I...>::value, int> = 0>
#endif
    constexpr auto operator()(I&&...i) const &&
    {
      const auto indices = std::array<std::size_t, sizeof...(I)> {static_cast<std::size_t>(std::forward<I>(i))...};
      return get_component(static_cast<const Derived&&>(*this), indices);
    }


    /**
     * \brief Access the component at index i
     * \return If <code>writable_by_component<Derived></code>, the component can be directly assigned.
     */
#ifdef __cpp_lib_concepts
    template<index_value I> requires (index_count_v<Derived> == dynamic_size or 1 >= index_count_v<Derived>) and
      internal::static_indices_within_bounds<Derived, I>::value
#else
    template<typename I, std::enable_if_t<(index_value<I>) and
      (index_count<Derived>::value == dynamic_size or 1 >= index_count<Derived>::value) and
      internal::static_indices_within_bounds<Derived, I>::value, int> = 0>
#endif
    constexpr auto operator[](I&& i) &
    {
      const auto indices = std::array<std::size_t, 1> {static_cast<std::size_t>(std::forward<I>(i))};
      if constexpr (writable_by_component<Derived, std::array<std::size_t, 1>>)
        return ElementAccessor(static_cast<Derived&>(*this), indices);
      else return get_component(static_cast<Derived&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value I> requires (index_count_v<Derived> == dynamic_size or 1 >= index_count_v<Derived>) and
      internal::static_indices_within_bounds<Derived, I>::value
#else
    template<typename I, std::enable_if_t<(index_value<I>) and
      (index_count<Derived>::value == dynamic_size or 1 >= index_count<Derived>::value) and
      internal::static_indices_within_bounds<Derived, I>::value, int> = 0>
#endif
    constexpr auto operator[](I&& i) &&
    {
      const auto indices = std::array<std::size_t, 1> {static_cast<std::size_t>(std::forward<I>(i))};
      if constexpr (writable_by_component<Derived, std::array<std::size_t, 1>>)
        return ElementAccessor(static_cast<Derived&&>(*this), indices);
      else return get_component(static_cast<Derived&&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value I> requires (index_count_v<Derived> == dynamic_size or 1 >= index_count_v<Derived>) and
      internal::static_indices_within_bounds<Derived, I>::value
#else
    template<typename I, std::enable_if_t<(index_value<I>) and
      (index_count<Derived>::value == dynamic_size or 1 >= index_count<Derived>::value) and
      internal::static_indices_within_bounds<Derived, I>::value, int> = 0>
#endif
    constexpr auto operator[](I&& i) const &
    {
      const auto indices = std::array<std::size_t, 1> {static_cast<std::size_t>(std::forward<I>(i))};
      return get_component(static_cast<const Derived&>(*this), indices);
    }


    /// \overload
#ifdef __cpp_lib_concepts
    template<index_value I> requires (index_count_v<Derived> == dynamic_size or 1 >= index_count_v<Derived>) and
      internal::static_indices_within_bounds<Derived, I>::value
#else
    template<typename I, std::enable_if_t<(index_value<I>) and
      (index_count<Derived>::value == dynamic_size or 1 >= index_count<Derived>::value) and
      internal::static_indices_within_bounds<Derived, I>::value, int> = 0>
#endif
    constexpr auto operator[](I&& i) const &&
    {
      const auto indices = std::array<std::size_t, 1> {static_cast<std::size_t>(std::forward<I>(i))};
      return get_component(static_cast<const Derived&&>(*this), indices);
    }

  private:

    NestedMatrix m_arg; //< The nested matrix.

  };

}

#endif //OPENKALMAN_BASEMATRIX_HPP

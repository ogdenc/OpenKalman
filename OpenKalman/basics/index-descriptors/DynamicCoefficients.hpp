/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the DynamicCoefficients class.
 */

#ifndef OPENKALMAN_DYNAMICCOEFFICIENTS_HPP
#define OPENKALMAN_DYNAMICCOEFFICIENTS_HPP

#include <vector>
#include <tuple>
#include <typeindex>

namespace OpenKalman
{
  /**
   * \brief A list of dynamic index descriptors that can be defined or extended at runtime.
   * \details At compile time, the structure is treated if it has zero dimension.
   * \tparam Scalar A scalar type associated with the index.
   */
  template<typename Scalar>
  struct DynamicCoefficients : AbstractDynamicTypedIndexDescriptor<Scalar>
  {
    /**
     * \brief Default constructor.
     */
    DynamicCoefficients() = default;


    /**
     * \brief Constructor taking any number of \ref index_descriptor objects Cs.
     * \details Index descriptors Cs can be either fixed or dynamic. If dynamic, the descriptor must be either
     * untyped or the same type as this DynamicCoefficients.
     * \tparam Cs A list of \ref index_descriptor objects.
     */
#ifdef __cpp_concepts
    template<index_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::same_as<Cs, DynamicCoefficients> and ...)) and
      (sizeof...(Cs) <= 1 or
        ((fixed_index_descriptor<Cs> or untyped_index_descriptor<Cs> or std::same_as<Cs, DynamicCoefficients>) and ...))
#else
    template<typename...Cs, std::enable_if_t<(index_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_same_v<Cs, DynamicCoefficients> and ...)) and
      (sizeof...(Cs) <= 1 or
        ((fixed_index_descriptor<Cs> or untyped_index_descriptor<Cs> or std::is_same_v<Cs, DynamicCoefficients>) and ...)),
      int> = 0>
#endif
    explicit DynamicCoefficients(Cs&&...cs)
    {
      dynamic_types.reserve((0 + ... + get_index_descriptor_component_count_of(cs)));
      index_table.reserve((0 + ... + get_dimension_size_of(cs)));
      euclidean_index_table.reserve((0 + ... + get_euclidean_dimension_size_of(cs)));
      fill_tables(0, 0, 0, std::forward<Cs>(cs)...);
    }


#ifdef __cpp_concepts
    template<index_descriptor...Cs> requires
      ((fixed_index_descriptor<Cs> or untyped_index_descriptor<Cs> or std::same_as<Cs, DynamicCoefficients>) and ...)
#else
    template<typename...Cs, std::enable_if_t<(index_descriptor<Cs> and ...) and
      ((fixed_index_descriptor<Cs> or untyped_index_descriptor<Cs> or std::is_same_v<Cs, DynamicCoefficients>) and ...), int> = 0>
#endif
    constexpr void
    extend(Cs&&...cs)
    {
      if constexpr (sizeof...(Cs) > 0) if (auto N = (0 + ... + get_index_descriptor_component_count_of(cs)); N > 1)
      {
        if (dynamic_types.capacity() < dynamic_types.size() + N)
          dynamic_types.reserve(dynamic_types.capacity() * 2);
        if (index_table.capacity() < (index_table.size() + ... + get_dimension_size_of(cs)))
          index_table.reserve(index_table.capacity() * 2);
        if (euclidean_index_table.capacity() < (euclidean_index_table.size() + ... + get_euclidean_dimension_size_of(cs)))
          euclidean_index_table.reserve(euclidean_index_table.capacity() * 2);
      }
      fill_tables(index_table.size(), euclidean_index_table.size(), dynamic_types.size(), std::forward<Cs>(cs)...);
    }


    using Getter = std::function<Scalar(std::size_t)>;

    using Setter = std::function<void(Scalar, std::size_t)>;


    /**
     * \brief Return a functor mapping elements of a matrix or other tensor to coordinates in Euclidean space.
     * \returns A functor returning the result of type <code>Scalar</code> and taking the following:
     * - an element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>);
     * - a local index relative to the Euclidean-transformed coordinates (starting at 0); and
     * - the starting location of the index descriptor within any larger set of index type descriptors
     */
    [[nodiscard]] virtual std::function<Scalar(const Getter&, std::size_t euclidean_local_index, std::size_t start)>
    to_euclidean_element() const final
    {
      return [&](const Getter& g, std::size_t euclidean_local_index, std::size_t start) {
        auto [tp, comp_euclidean_local_index, comp_start] = euclidean_index_table[euclidean_local_index];
        return dynamic_types[tp].get().to_euclidean_element()(g, comp_euclidean_local_index, start + comp_start);
      };
    }


    /**
     * \brief Get a coordinate in Euclidean space corresponding to a coefficient in a matrix with typed coefficients.
     * \param t A tensor or matrix
     * \param i The first index of the matrix
     * \param is Other indices of the matrix
     * \return The scalar value of the transformed coordinate in Euclidean space corresponding to the provided index i
     */
#ifdef __cpp_concepts
    template<indexible T, std::convertible_to<const std::size_t>...Is>
    requires (sizeof...(Is) <= max_indices_of_v<T> - 1) and element_gettable<T, std::size_t, Is...>
#else
    template<typename T, typename...Is, std::enable_if_t<indexible<T> and
      (std::is_convertible<Is, const std::size_t>::value and ...) and sizeof...(Is) <= max_indices_of_v<T> - 1, int> = 0>
#endif
    auto to_euclidean_element(const T& t, std::size_t i, Is...is)
    {
      auto [tp, euclidean_local_index, start] = euclidean_index_table[i];
      Getter g {[&t, is...](std::size_t ix) { return get_element(t, ix, is...); }};
      return dynamic_types[tp].get().to_euclidean_element()(g, euclidean_local_index, start);
    }


    /**
     * \brief Return a functor mapping coordinates in Euclidean space to elements of a matrix or other tensor.
     * \returns A functor returning the result of type <code>Scalar</code> and taking the following:
     * - an element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>);
     * - a local index relative to this index descriptor (starting at 0) accessing the element; and
     * - the starting location within any larger set of Euclidean-transformed index type descriptors
     */
    virtual std::function<Scalar(const Getter&, std::size_t local_index, std::size_t euclidean_start)>
    from_euclidean_element() const final
    {
      return [&](const Getter& g, std::size_t local_index, std::size_t euclidean_start) {
        auto [tp, comp_local_index, comp_euclidean_start] = index_table[local_index];
        return dynamic_types[tp].get().from_euclidean_element()(g, comp_local_index, euclidean_start + comp_euclidean_start);
      };
    }


    /**
     * \brief Get a typed coefficient corresponding to its corresponding coordinate in Euclidean space.
     * \param t A tensor or matrix
     * \param i The first index of the Euclidean-transformed matrix
     * \param is Other indices of the matrix
     * \return The scalar value of the typed coefficient corresponding to the provided index i
     */
#ifdef __cpp_concepts
    template<indexible T, std::convertible_to<const std::size_t>...Is>
    requires (sizeof...(Is) <= max_indices_of_v<T> - 1) and element_gettable<T, std::size_t, Is...>
#else
    template<typename T, typename...Is, std::enable_if_t<indexible<T> and
      (std::is_convertible<Is, const std::size_t>::value and ...) and sizeof...(Is) <= max_indices_of_v<T> - 1, int> = 0>
#endif
    auto from_euclidean_element(const T& t, std::size_t i, Is...is) const
    {
      auto [tp, local_index, euclidean_start] = index_table[i];
      Getter g {[&t, is...](std::size_t ix) { return get_element(t, ix, is...); }};
      return dynamic_types[tp].get().from_euclidean_element()(g, local_index, euclidean_start);
    }


    /**
     * \brief Return a functor wrapping an angle.
     * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
     * \returns A functor returning the result of type <code>Scalar</code> and taking the following:
     * - an element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>);
     * - a local index relative to this index descriptor (starting at 0) accessing the element; and
     * - the starting location of the index descriptor within any larger set of index type descriptors
     */
    virtual std::function<Scalar(const Getter&, std::size_t local_index, std::size_t start)>
    wrap_get_element() const final
    {
      return [&](const Getter& g, std::size_t local_index, std::size_t start) {
        auto [tp, comp_local_index, comp_start] = index_table[local_index];
        return dynamic_types[tp].get().from_euclidean_element()(g, comp_local_index, start + comp_start);
      };
    }


    /**
     * \brief Wrap a given coefficient and return its wrapped, scalar value.
     * \param t A tensor or matrix
     * \param i The first index of the matrix
     * \param is Other indices of the matrix
     * \return The scalar value of the wrapped coefficient corresponding to the provided index i
     */
#ifdef __cpp_concepts
    template<indexible T, std::convertible_to<const std::size_t>...Is>
    requires (sizeof...(Is) <= max_indices_of_v<T> - 1) and element_gettable<T, std::size_t, Is...>
#else
    template<typename T, typename...Is, std::enable_if_t<indexible<T> and
      (std::is_convertible<Is, const std::size_t>::value and ...) and sizeof...(Is) <= max_indices_of_v<T> - 1, int> = 0>
#endif
    auto wrap_get_element(const T& t, std::size_t i, Is...is) const
    {
      auto [tp, local_index, start] = index_table[i];
      Getter g {[&t, is...](std::size_t ix) { return get_element(t, ix, is...); }};
      return dynamic_types[tp].get().wrap_get_element()(g, local_index, start);
    }


    /**
     * \brief Return a functor setting an angle and then wrapping the angle.
     * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
     * \returns A functor returning the result of type <code>Scalar</code> and taking the following:
     * - an element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>);
     * - an element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>);
     * - a new value of type Scalar to set;
     * - a local index relative to this index descriptor (starting at 0) accessing the element; and
     * - the starting location of the index descriptor within any larger set of index type descriptors
     */
    virtual std::function<void(const Setter&, const Getter&, Scalar x, std::size_t local_index, std::size_t start)>
    wrap_set_element() const final
    {
      return [&](const Setter& s, const Getter& g, Scalar x, std::size_t local_index, std::size_t start) {
        auto [tp, comp_local_index, comp_start] = index_table[local_index];
        return dynamic_types[tp].get().wrap_set_element()(s, g, x, comp_local_index, start + comp_start);
      };
    }


    /**
     * \brief Set the scalar value of a given typed coefficient in a matrix, and wrap the matrix column.
     * \param t A tensor or matrix
     * \param x The new value to be set
     * \param i The first index of the matrix
     * \param is Other indices of the matrix
     */
#ifdef __cpp_concepts
    template<indexible T, std::convertible_to<const std::size_t>...Is>
    requires (sizeof...(Is) <= max_indices_of_v<T> - 1) and element_gettable<T, std::size_t, Is...>
#else
    template<typename T, typename...Is, std::enable_if_t<indexible<T> and
      (std::is_convertible<Is, const std::size_t>::value and ...) and sizeof...(Is) <= max_indices_of_v<T> - 1, int> = 0>
#endif
    void wrap_set_element(T& t, const scalar_type_of_t<T> x, const std::size_t i, Is...is) const
    {
      auto [tp, local_index, start] = index_table[i];
      Setter s {[&t, is...](Scalar x, std::size_t i) { return set_element(t, x, i, is...); }};
      Getter g {[&t, is...](std::size_t i) { return get_element(t, i, is...); }};
      return dynamic_types[tp].get().wrap_set_element()(s, g, x, local_index, start);
    }


    [[nodiscard]] bool is_untyped() const final
    {
      for (auto i = dynamic_types.begin(); i != dynamic_types.end(); ++i)
        if (not i->get().is_untyped()) return false;
      return true;
    }


    /**
     * \brief Whether this index descriptor is composite
     * \sa composite_index_descriptor
     */
    bool is_composite() const final
    {
      return true;
    }

  private:

    // ---------- tables ---------- //

    using dynamic_types_t = std::vector<std::reference_wrapper<const AbstractDynamicTypedIndexDescriptor<Scalar>>>;

    dynamic_types_t dynamic_types;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> index_table;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> euclidean_index_table;

    // ---------- extending tables ---------- //

    /*
     * \internal
     * \tparam euclidean Whether the relevant vector is in Euclidean space (true) or not (false)
     * \tparam local_index The local index for indices associated with each type (resets to 0 when t increments)
     * \tparam C The current index type being processed
     * \tparam Cs Remaining index types to process
     * \param i The row index
     * \param t The index (within dynamic_types) of C
     * \param start The start location in the corresponding euclidean or non-euclidean vector
     * \return A tuple of tuples of {t, local_index, start}
     */
    template<bool euclidean, std::size_t local_index, typename C, typename...Cs>
    constexpr void extend_table_fixed(std::size_t i, std::size_t t, std::size_t start)
    {
      constexpr auto i_size = dimension_size_of_v<C>;
      constexpr auto e_size = euclidean_dimension_size_of_v<C>;
      if constexpr (local_index >= (euclidean ? e_size : i_size))
      {
        extend_table_fixed<euclidean, 0, Cs...>(i, t + 1, start + (euclidean ? i_size : e_size));
      }
      else
      {
        (euclidean ? euclidean_index_table : index_table).emplace_back(t, local_index, start);
        extend_table_fixed<euclidean, local_index + 1, C, Cs...>(i + 1, t, start);
      }
    }


    // \overload
    template<bool euclidean, std::size_t local_index>
    constexpr void extend_table_fixed(std::size_t i, std::size_t t, std::size_t start) {}


    template<typename...Cs>
    constexpr void fill_tables_fixed(std::size_t i, std::size_t i_e, std::size_t t, const Coefficients<Cs...>&)
    {
      (dynamic_types.emplace_back(std::cref(DynamicTypedIndexDescriptor<Scalar, Cs>::get_instance())), ...);
      extend_table_fixed<false, 0, Cs...>(i, t, i_e);
      extend_table_fixed<true, 0, Cs...>(i_e, t, i);
    }


    template<typename C, typename...Cs>
    constexpr void fill_tables_dynamic(std::size_t i, std::size_t i_e, std::size_t t, C&& c, Cs&&...cs)
    {
      if constexpr (untyped_index_descriptor<C>)
      {
        // \todo Add untyped index descriptors without breaking them up.
        auto N = t + get_dimension_size_of(c);
        for (; t < N; ++i, ++i_e, ++t)
        {
          dynamic_types.emplace_back(std::cref(DynamicTypedIndexDescriptor<Scalar, Axis>::get_instance()));
          index_table.emplace_back(t, 0, i_e);
          euclidean_index_table.emplace_back(t, 0, i);
        }
        fill_tables(i, i_e, t, std::forward<Cs>(cs)...);
      }
      else // C is DynamicCoefficients<T>.
      {
        auto new_i = i + c.index_table.size();
        auto new_i_e = i_e + c.euclidean_index_table.size();
        auto new_t = t + c.dynamic_types.size();

        for (auto&& j : c.dynamic_types)
          dynamic_types.emplace_back(std::forward<decltype(j)>(j));

        for (auto&& j : c.index_table)
          index_table.emplace_back(
            t + std::get<0>(std::forward<decltype(j)>(j)),
            std::get<1>(std::forward<decltype(j)>(j)),
            i_e + std::get<2>(std::forward<decltype(j)>(j)));

        for (auto&& j : c.euclidean_index_table)
          euclidean_index_table.emplace_back(
            t + std::get<0>(std::forward<decltype(j)>(j)),
            std::get<1>(std::forward<decltype(j)>(j)),
            i + std::get<2>(std::forward<decltype(j)>(j)));

        fill_tables(new_i, new_i_e, new_t, std::forward<Cs>(cs)...);
      }
    }


    template<typename C, typename...Cs>
    constexpr void fill_tables(std::size_t i, std::size_t i_e, std::size_t t, C&& c, Cs&&...cs)
    {
      if constexpr (fixed_index_descriptor<C>)
      {
        using red_C = reduced_fixed_index_descriptor_t<C>;
        fill_tables_fixed(i, i_e, t, red_C {});
        fill_tables(i + dimension_size_of_v<C>, i_e + euclidean_dimension_size_of_v<C>,
          t + index_descriptor_components_of<red_C>::value, std::forward<Cs>(cs)...);
      }
      else // dynamic_index_descriptor<C>
      {
        fill_tables_dynamic(i, i_e, t, std::forward<C>(c), std::forward<Cs>(cs)...);
      }
    }


    constexpr void fill_tables(std::size_t, std::size_t, std::size_t) {}

    // ---------- comparison ---------- //

    template<typename C, typename...Cs, typename It>
    static constexpr bool compare_fixed_impl(It&& i)
    {
      if (std::type_index {typeid(i->get())} == std::type_index {typeid(DynamicTypedIndexDescriptor<Scalar, C>)})
        return compare_fixed_impl<Cs...>(++std::forward<It>(i));
      else
        return false;
    }


    template<typename It>
    static constexpr bool compare_fixed_impl(It&&)
    {
      return true;
    }


    template<typename...C>
    constexpr bool compare_fixed(const Coefficients<C...>&) const
    {
      return DynamicCoefficients::compare_fixed_impl<C...>(dynamic_types.begin());
    }


    /*
     * \internal
     * \brief Determine whether an index descriptor is equivalent to this.
     * \details This assumes that c.size() == size()
     * \return <code>true</code> if equivalent
     */
#ifdef __cpp_concepts
    template<index_descriptor C>
#else
    template<typename C, std::enable_if_t<index_descriptor<C>, int> = 0>
#endif
    constexpr bool
    is_equivalent(const C& c) const
    {
      if constexpr (fixed_index_descriptor<C>)
      {
        return compare_fixed(reduced_fixed_index_descriptor_t<C> {});
      }
      else if constexpr (untyped_index_descriptor<C>)
      {
        return is_untyped();
      }
      else // c is instance of DynamicCoefficients<U>
      {
        // \todo Do a more sophisticated comparison between composite index descriptors.
        for (auto i = dynamic_types.begin(), j = c.dynamic_types.begin(); j != c.dynamic_types.end(); ++i, ++j)
        {
          if (std::type_index {typeid(i->get())} != std::type_index {typeid(j->get())}) return false;
        }
        return true;
      }
    }

    // ---------- friends ---------- //

    template<typename U> friend struct dimension_size_of;
    template<typename U> friend struct euclidean_dimension_size_of;
    template<typename U> friend struct index_descriptor_components_of;

#ifdef __cpp_impl_three_way_comparison
    template<index_descriptor A, index_descriptor B> requires (not std::integral<A>) and (not std::integral<B>)
    friend constexpr auto operator<=>(const A&, const B&);
#else
    template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
        (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int>>
    friend constexpr bool operator==(const A& a, const B& b);

    template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
      (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int>>
    friend constexpr bool operator<(const A& a, const B& b);

    template<typename A, typename B, std::enable_if_t<index_descriptor<A> and index_descriptor<B> and
      (not std::is_integral_v<A>) and (not std::is_integral_v<B>), int>>
    friend constexpr bool operator>(const A& a, const B& b);
#endif

  };


  /**
    * \internal
    * \brief The size of DynamicCoefficients is not known at compile time.
    */
   template<typename T>
   struct dimension_size_of<DynamicCoefficients<T>> : std::integral_constant<std::size_t, dynamic_size>
   {
     static std::size_t get(const DynamicCoefficients<T>& t) { return t.index_table.size(); }
   };


  /**
    * \internal
    * \brief The Euclidean size of DynamicCoefficients is not known at compile time.
    */
   template<typename T>
   struct euclidean_dimension_size_of<DynamicCoefficients<T>> : std::integral_constant<std::size_t, dynamic_size>
   {
     static std::size_t get(const DynamicCoefficients<T>& t) { return t.euclidean_index_table.size(); }
   };


    /**
     * \brief The number of atomic components.
     */
    template<typename T>
    struct index_descriptor_components_of<DynamicCoefficients<T>> : std::integral_constant<std::size_t, dynamic_size>
    {
      constexpr static std::size_t get(const DynamicCoefficients<T>& t) { return t.dynamic_types.size(); }
    };


  /**
   * \internal
   * \brief The difference type for DynamicCoefficients is also DynamicCoefficients
   */
  template<typename T>
  struct dimension_difference_of<DynamicCoefficients<T>> { using type = DynamicCoefficients<T>; };


} // namespace OpenKalman


#endif //OPENKALMAN_DYNAMICCOEFFICIENTS_HPP

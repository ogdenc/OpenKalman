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
 * \brief Definition of the DynamicTypedIndex class.
 */

#ifndef OPENKALMAN_DYNAMICTYPEDINDEX_HPP
#define OPENKALMAN_DYNAMICTYPEDINDEX_HPP

#include <vector>
#include <tuple>
#include <complex>
#include <typeindex>

namespace OpenKalman
{
  /**
   * \brief A dynamic list of atomic index descriptor objects that can be defined or extended at runtime.
   * \details At compile time, the structure is treated if it has zero dimension.
   * \tparam AllowableScalarTypes The allowable scalar types for elements associated with this index descriptor.
   */
#ifdef __cpp_concepts
  template<floating_scalar_type...AllowableScalarTypes>
#else
  template<typename...AllowableScalarTypes>
#endif
  struct DynamicTypedIndex
  {
  private:
#ifndef __cpp_concepts
    static_assert((floating_scalar_type<AllowableScalarTypes> and ...));
#endif

#ifdef __cpp_concepts
    template<floating_scalar_type...>
#else
    template<typename...>
#endif
    friend struct DynamicTypedIndex;


    using AtomicType = std::conditional_t<sizeof...(AllowableScalarTypes) == 0,
      internal::AnyAtomicIndexDescriptor<double, float, long double,
        std::complex<double>, std::complex<float>, std::complex<long double>>,
      internal::AnyAtomicIndexDescriptor<AllowableScalarTypes...>>;

  public:

    /**
     * \brief Default constructor.
     */
    DynamicTypedIndex() = default;


    /**
     * \brief Constructor taking any number of \ref index_descriptor objects Cs.
     * \details Index descriptors Cs can be either fixed or dynamic. If dynamic, the descriptor must be either
     * untyped or the same type as this DynamicTypedIndex.
     * \tparam Cs A list of \ref index_descriptor objects.
     */
#ifdef __cpp_concepts
    template<index_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::same_as<Cs, DynamicTypedIndex> and ...)) and
      (sizeof...(Cs) == 1 or ((fixed_index_descriptor<Cs> or dynamic_index_descriptor<Cs>) and ...))
#else
    template<typename...Cs, std::enable_if_t<(index_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_same_v<Cs, DynamicTypedIndex> and ...)) and
      (sizeof...(Cs) == 1 or ((fixed_index_descriptor<Cs> or dynamic_index_descriptor<Cs>) and ...)),
      int> = 0>
#endif
    explicit DynamicTypedIndex(Cs&&...cs)
    {
      dynamic_types.reserve((0 + ... + get_index_descriptor_component_count_of(cs)));
      index_table.reserve((0 + ... + get_dimension_size_of(cs)));
      euclidean_index_table.reserve((0 + ... + get_euclidean_dimension_size_of(cs)));
      fill_tables(0, 0, 0, std::forward<Cs>(cs)...);
    }


    /**
     * \brief Constructor taking another DynamicTypedIndex and other \ref index_descriptor objects.
     * \details This forwards to the copy or move constructors and then extends by the additional index descriptors.
     * \param c A DynamicTypedIndex object
     * \param cn A first \ref index_descriptor.
     * \param cs A list of \ref index_descriptor objects.
     */
#ifdef __cpp_concepts
    DynamicTypedIndex(auto&& c, index_descriptor auto&& c0, index_descriptor auto&&...cn)
    requires std::same_as<std::decay_t<decltype(c)>, DynamicTypedIndex>
#else
    template<typename C, typename C0, typename...Cn, std::enable_if_t<
      std::is_same_v<std::decay_t<C>, DynamicTypedIndex> and (index_descriptor<Cn> and ...), int> = 0>
    DynamicTypedIndex(C&& c, C0&& c0, Cn&&...cn)
#endif
      : DynamicTypedIndex(std::forward<decltype(c)>(c))
    {
      extend(std::forward<decltype(c0)>(c0), std::forward<decltype(cn)>(cn)...);
    }


    /**
     * \brief Extend the DynamicTypedIndex by appending to the end of the list.
     * \tparam Cs One or more \ref index_descriptor objects
     */
  #ifdef __cpp_concepts
    template<index_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      ((fixed_index_descriptor<Cs> or euclidean_index_descriptor<Cs> or std::same_as<Cs, DynamicTypedIndex>) and ...)
  #else
    template<typename...Cs, std::enable_if_t<(index_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      ((fixed_index_descriptor<Cs> or euclidean_index_descriptor<Cs> or std::is_same_v<Cs, DynamicTypedIndex>) and ...), int> = 0>
  #endif
    DynamicTypedIndex& extend(Cs&&...cs)
    {
      if (auto N = (0 + ... + get_index_descriptor_component_count_of(cs)); N > 1)
      {
        if (dynamic_types.capacity() < dynamic_types.size() + N)
          dynamic_types.reserve(dynamic_types.capacity() * 2);
        if (index_table.capacity() < (index_table.size() + ... + get_dimension_size_of(cs)))
          index_table.reserve(index_table.capacity() * 2);
        if (euclidean_index_table.capacity() < (euclidean_index_table.size() + ... + get_euclidean_dimension_size_of(cs)))
          euclidean_index_table.reserve(euclidean_index_table.capacity() * 2);
      }
      fill_tables(index_table.size(), euclidean_index_table.size(), dynamic_types.size(), std::forward<Cs>(cs)...);
      return *this;
    }

    // ---------- iterators ---------- //

    /**
     * \return Iterator marking the beginning of a vector containing a set of DynamicTypedIndexDescriptor objects.
     */
    auto begin() const { return dynamic_types.begin(); }


    /**
     * \return Iterator marking the end of a vector containing a set of DynamicTypedIndexDescriptor objects.
     */
    auto end() const { return dynamic_types.end(); }

  private:

    // ---------- extending tables ---------- //

    /*
     * \internal
     * \tparam euclidean Whether the relevant vector is in Euclidean space (true) or not (false)
     * \tparam local_index The local index for indices associated with each of dynamic_types (resets to 0 when t increments)
     * \tparam C The current index type being processed
     * \tparam Cs Remaining index types to process
     * \param i The row index
     * \param t The component index within dynamic_types
     * \param start The start location in the corresponding euclidean or non-euclidean vector
     * \return A tuple of tuples of {t, local_index, start}
     */
    template<bool euclidean, std::size_t local_index, typename C, typename...Cs>
    void extend_table_fixed(std::size_t i, std::size_t t, std::size_t start)
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
    void extend_table_fixed(std::size_t i, std::size_t t, std::size_t start) {}


    template<typename...Cs>
    void fill_tables_fixed(std::size_t i, std::size_t i_e, std::size_t t, const TypedIndex<Cs...>&)
    {
      (dynamic_types.emplace_back(AtomicType {Cs{}}), ...);
      extend_table_fixed<false, 0, Cs...>(i, t, i_e);
      extend_table_fixed<true, 0, Cs...>(i_e, t, i);
    }


    template<typename C, typename...Cs>
    void fill_tables_dynamic(std::size_t i, std::size_t i_e, std::size_t t, C&& c, Cs&&...cs)
    {
      if constexpr (euclidean_index_descriptor<C>)
      {
        // \todo Add untyped index descriptors without breaking them up.
        auto N = t + get_dimension_size_of(c);
        for (; t < N; ++i, ++i_e, ++t)
        {
          dynamic_types.emplace_back(AtomicType {Axis{}});
          index_table.emplace_back(t, 0, i_e);
          euclidean_index_table.emplace_back(t, 0, i);
        }
        fill_tables(i, i_e, t, std::forward<Cs>(cs)...);
      }
      else // C is DynamicTypedIndex<T>.
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
    void fill_tables(std::size_t i, std::size_t i_e, std::size_t t, C&& c, Cs&&...cs)
    {
      if constexpr (fixed_index_descriptor<C>)
      {
        using red_C = canonical_fixed_index_descriptor_t<std::decay_t<C>>;
        fill_tables_fixed(i, i_e, t, red_C {});
        fill_tables(i + dimension_size_of_v<C>, i_e + euclidean_dimension_size_of_v<C>,
          t + index_descriptor_components_of<red_C>::value, std::forward<Cs>(cs)...);
      }
      else // dynamic_index_descriptor<C>
      {
        fill_tables_dynamic(i, i_e, t, std::forward<C>(c), std::forward<Cs>(cs)...);
      }
    }


    void fill_tables(std::size_t, std::size_t, std::size_t) {}


    // ---------- comparison ---------- //

    template<typename It, typename EndIt, std::size_t N>
    static constexpr bool partial_compare(It it, EndIt endit, const Dimensions<N>&, TypedIndex<>)
    {
      return it == endit or it->is_euclidean();
    }

    template<typename It, typename EndIt, std::size_t N, typename C, typename...Cs>
    static constexpr bool partial_compare(It it, EndIt endit, const Dimensions<N>& d, TypedIndex<C, Cs...>)
    {
      if (it->is_euclidean())
      {
        std::size_t it_size = it->size();
        std::size_t d_size = get_dimension_size_of(d);
        if (it_size == d_size)
          return partial_compare(++it, endit, TypedIndex<C, Cs...> {});
        else if (it_size < d_size)
          return partial_compare(++it, endit, Dimensions {static_cast<std::size_t>(d_size - it_size)}, TypedIndex<C, Cs...> {});
        else // it_size > d_size
        {
          if constexpr (euclidean_index_descriptor<C>)
            return partial_compare(it, endit, Dimensions {d_size + dimension_size_of_v<C>}, TypedIndex<Cs...> {});
          else
            return false;
        }
      }
      else return false;
    }

    template<typename It, typename EndIt>
    static constexpr bool partial_compare(It, EndIt, TypedIndex<>) { return true; }

    template<typename It, typename EndIt, typename C, typename...Cs>
    static constexpr bool partial_compare(It it, EndIt endit, TypedIndex<C, Cs...>)
    {
      if (it == endit) return true;
      else
      {
        if constexpr (euclidean_index_descriptor<C>)
        {
          return partial_compare(it, endit, Dimensions {dimension_size_of_v<C>}, TypedIndex<Cs...> {});
        }
        else
        {
          if (it->get_type_index() == std::type_index {typeid(C)}) return partial_compare(++it, endit, TypedIndex<Cs...> {});
          else return false;
        }
      }
    }


  #ifdef __cpp_concepts
    template<typename Arg> requires fixed_index_descriptor<Arg> or euclidean_index_descriptor<Arg>
  #else
    template<typename Arg, std::enable_if_t<fixed_index_descriptor<Arg> or euclidean_index_descriptor<Arg>, int> = 0>
  #endif
    bool partial_match(const Arg& arg) const
    {
      if constexpr (fixed_index_descriptor<Arg>)
        return partial_compare(dynamic_types.begin(), dynamic_types.end(), canonical_fixed_index_descriptor_t<Arg> {});
      else
        return partial_compare(dynamic_types.begin(), dynamic_types.end(), Dimensions<dynamic_size>(get_dimension_size_of(arg)), TypedIndex<> {});
    }


    template<typename...S>
    bool partial_match(const DynamicTypedIndex<S...>& arg) const
    {
      // \todo Do a more sophisticated comparison.
      auto i = begin();
      for (auto j = arg.begin(); i != end() and j != arg.end(); ++i, ++j)
      {
        if (i->get_type_index() != j->get_type_index()) return false;
      }
      return true;
    }


  #if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
    template<index_descriptor A, index_descriptor B>
    friend constexpr auto operator<=>(const A& a, const B& b) requires (not std::integral<A>) and (not std::integral<B>);
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

  public:

    // ---------- operator+= is the same as extend ---------- //

  #ifdef __cpp_concepts
    template<index_descriptor B>
  #else
    template<typename B>
  #endif
    constexpr decltype(auto) operator+=(B&& b) { return extend(std::forward<B>(b)); }


    // ---------- friends ---------- //

  friend struct interface::DynamicIndexDescriptorTraits<DynamicTypedIndex>;


  private:

    // ---------- tables ---------- //

    std::vector<AtomicType> dynamic_types;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> index_table;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> euclidean_index_table;

  };


  // -------------------------------------- //
  //            deduction guides            //
  // -------------------------------------- //

  /**
   * \brief Deduce automatic scalar type for a constructor that does not specify one.
   * \tparam Cs Constructor arguments
   */
  template<typename...Cs>
  DynamicTypedIndex(Cs&&...) -> DynamicTypedIndex<>;

  /**
   * \brief Deduce scalar type when the constructor's first argument is a DynamicTypedIndex.
   */
  template<typename...S, typename C, typename...Cs>
  DynamicTypedIndex(DynamicTypedIndex<S...>&&, C&&, Cs&&...) -> DynamicTypedIndex<S...>;

  /// \overload
  template<typename...S, typename C, typename...Cs>
  DynamicTypedIndex(const DynamicTypedIndex<S...>&, C&&, Cs&&...) -> DynamicTypedIndex<S...>;


  namespace detail
  {
    template<typename Scalar, typename>
    struct is_allowable_scalar : std::false_type {};

    template<typename Scalar, typename...S>
    struct is_allowable_scalar<Scalar, internal::AnyAtomicIndexDescriptor<S...>>
      : std::bool_constant<(std::is_same_v<std::decay_t<Scalar>, S> or ...)> {};


    /**
     * \internal
     * \brief Tests whether Scalar is an allowable scalar type.
     * \tparam Scalar A \ref floating_scalar_type
     * \tparam AtomicType An object of type internal::AnyAtomicIndexDescriptor
     */
#ifdef __cpp_concepts
    template<typename Scalar, typename AtomicType>
    concept allowable_scalar =
#else
    template<typename Scalar, typename AtomicType>
    constexpr bool allowable_scalar =
#endif
      floating_scalar_type<Scalar> and is_allowable_scalar<Scalar, std::decay_t<AtomicType>>::value;

  } // namespace detail


  // --------- //
  //   traits  //
  // --------- //

  namespace interface
  {
    /**
     * \internal
     * \brief traits for DynamicTypedIndex.
     */
    template<typename...AllowableScalarTypes>
    struct DynamicIndexDescriptorTraits<DynamicTypedIndex<AllowableScalarTypes...>>
    {
    private:

      using AtomicType = typename DynamicTypedIndex<AllowableScalarTypes...>::AtomicType;

    public:

      explicit constexpr DynamicIndexDescriptorTraits(const DynamicTypedIndex<AllowableScalarTypes...>& t)
        : m_descriptor {t} {};

      constexpr std::size_t get_size() const { return m_descriptor.index_table.size(); }

      constexpr std::size_t get_euclidean_size() const { return m_descriptor.euclidean_index_table.size(); }

      constexpr std::size_t get_component_count() const { return m_descriptor.dynamic_types.size(); }

      constexpr bool is_euclidean() const
      {
        for (auto i = m_descriptor.dynamic_types.begin(); i != m_descriptor.dynamic_types.end(); ++i)
          if (not i->is_euclidean()) return false;
        return true;
      }


  #ifdef __cpp_concepts
      floating_scalar_type auto
      to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start) const
      requires requires (std::size_t i){ {g(i)} -> detail::allowable_scalar<AtomicType>; }
  #else
      template<typename G, std::enable_if_t<
        detail::allowable_scalar<typename std::invoke_result<const G&, std::size_t>::type, AtomicType>, int> = 0>
      auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start) const
  #endif
      {
        auto [tp, comp_euclidean_local_index, comp_start] = m_descriptor.euclidean_index_table[euclidean_local_index];
        return m_descriptor.dynamic_types[tp].to_euclidean_element(g, comp_euclidean_local_index, start + comp_start);
      }


      /**
       * \brief Maps a coordinate in Euclidean space to an element.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index relative to the original coordinates (starting at 0)
       * \param start The starting index within the Euclidean-transformed indices
       */
  #ifdef __cpp_concepts
      floating_scalar_type auto
      from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start) const
      requires requires (std::size_t i){ {g(i)} -> detail::allowable_scalar<AtomicType>; }
  #else
      template<typename G, std::enable_if_t<
        detail::allowable_scalar<typename std::invoke_result<const G&, std::size_t>::type, AtomicType>, int> = 0>
      auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start) const
  #endif
      {
        auto [tp, comp_local_index, comp_euclidean_start] = m_descriptor.index_table[local_index];
        return m_descriptor.dynamic_types[tp].from_euclidean_element(g, comp_local_index, euclidean_start + comp_euclidean_start);
      }


      /**
       * \brief Perform modular wrapping of an element.
       * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index relative to the original coordinates (starting at 0)
       * \param start The starting location of the angle within any larger set of index type descriptors
       */
  #ifdef __cpp_concepts
        floating_scalar_type auto
        wrap_get_element(const auto& g, std::size_t local_index, std::size_t start) const
        requires requires (std::size_t i){ {g(i)} -> detail::allowable_scalar<AtomicType>; }
  #else
        template<typename G, std::enable_if_t<
          detail::allowable_scalar<typename std::invoke_result<const G&, std::size_t>::type, AtomicType>, int> = 0>
        auto wrap_get_element(const G& g, std::size_t local_index, std::size_t start) const
  #endif
      {
        auto [tp, comp_local_index, comp_start] = m_descriptor.index_table[local_index];
        return m_descriptor.dynamic_types[tp].wrap_get_element(g, comp_local_index, start + local_index - comp_local_index);
      }


      /**
       * \brief Set an element and then perform any necessary modular wrapping.
       * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
       * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param x The scalar value to be set
       * \param local_index A local index relative to the original coordinates (starting at 0)
       * \param start The starting location of the angle within any larger set of index type descriptors
       */
  #ifdef __cpp_concepts
      void wrap_set_element(const auto& s, const auto& g,
        const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start) const
      requires requires (std::size_t i){ s(x, i); {x} -> detail::allowable_scalar<AtomicType>; }
  #else
      template<typename S, typename G, std::enable_if_t<
        std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value and
        detail::allowable_scalar<typename std::invoke_result<G, std::size_t>::type, AtomicType>, int> = 0>
      void wrap_set_element(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
        std::size_t local_index, std::size_t start) const
  #endif
      {
        auto [tp, comp_local_index, comp_start] = m_descriptor.index_table[local_index];
        return m_descriptor.dynamic_types[tp].wrap_set_element(s, g, x, comp_local_index, start + local_index - comp_local_index);
      }

    private:

      const DynamicTypedIndex<AllowableScalarTypes...>& m_descriptor;

    };


  } // namespace interface

} // namespace OpenKalman


#endif //OPENKALMAN_DYNAMICTYPEDINDEX_HPP

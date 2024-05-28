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
 * \brief Definition of the DynamicDescriptor class.
 */

#ifndef OPENKALMAN_DYNAMICDESCRIPTOR_HPP
#define OPENKALMAN_DYNAMICDESCRIPTOR_HPP

#include <vector>
#include <tuple>
#include <complex>
#include <typeindex>

namespace OpenKalman
{
  /**
   * \brief A dynamic list of \ref atomic_fixed_vector_space_descriptor objects that can be defined or extended at runtime.
   * \details At compile time, the structure is treated if it has zero dimension.
   * \tparam AllowableScalarTypes The allowable scalar types for elements associated with this object.
   */
#ifdef __cpp_concepts
  template<scalar_type...AllowableScalarTypes>
#else
  template<typename...AllowableScalarTypes>
#endif
  struct DynamicDescriptor
  {
  private:
#ifndef __cpp_concepts
    static_assert((scalar_type<AllowableScalarTypes> and ...));
#endif

#ifdef __cpp_concepts
    template<scalar_type...>
#else
    template<typename...>
#endif
    friend struct DynamicDescriptor;


    using AtomicType = std::conditional_t<sizeof...(AllowableScalarTypes) == 0,
      internal::AnyAtomicVectorSpaceDescriptor<double, float, long double,
        std::complex<double>, std::complex<float>, std::complex<long double>>,
      internal::AnyAtomicVectorSpaceDescriptor<AllowableScalarTypes...>>;

  public:

    /**
     * \brief Default constructor.
     */
    DynamicDescriptor() = default;


    /**
     * \brief Constructor taking any number of \ref vector_space_descriptor Cs.
     * \details \ref vector_space_descriptor objects Cs can be either fixed or dynamic. If dynamic, the vector_space_descriptor must be either
     * untyped or the same type as this DynamicDescriptor.
     * \tparam Cs A list of sets of \ref vector_space_descriptor for a set of indices.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::same_as<Cs, DynamicDescriptor> and ...)) and
      (sizeof...(Cs) == 1 or ((fixed_vector_space_descriptor<Cs> or dynamic_vector_space_descriptor<Cs>) and ...))
#else
    template<typename...Cs, std::enable_if_t<(vector_space_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_same_v<Cs, DynamicDescriptor> and ...)) and
      (sizeof...(Cs) == 1 or ((fixed_vector_space_descriptor<Cs> or dynamic_vector_space_descriptor<Cs>) and ...)),
      int> = 0>
#endif
    explicit DynamicDescriptor(Cs&&...cs)
    {
      dynamic_types.reserve((0 + ... + get_vector_space_descriptor_component_count_of(cs)));
      index_table.reserve((0 + ... + get_dimension_size_of(cs)));
      euclidean_index_table.reserve((0 + ... + get_euclidean_dimension_size_of(cs)));
      fill_tables(0, 0, 0, std::forward<Cs>(cs)...);
    }


    /**
     * \brief Constructor taking another DynamicDescriptor and other list of \ref vector_space_descriptor objects.
     * \details This forwards to the copy or move constructors and then extends by the additional \ref vector_space_descriptor.
     * \param c A DynamicDescriptor object
     * \param cn A first \ref vector_space_descriptor.
     * \param cs A list of \ref vector_space_descriptor objects.
     */
#ifdef __cpp_concepts
    DynamicDescriptor(auto&& c, vector_space_descriptor auto&& c0, vector_space_descriptor auto&&...cn)
    requires std::same_as<std::decay_t<decltype(c)>, DynamicDescriptor>
#else
    template<typename C, typename C0, typename...Cn, std::enable_if_t<
      std::is_same_v<std::decay_t<C>, DynamicDescriptor> and (vector_space_descriptor<Cn> and ...), int> = 0>
    DynamicDescriptor(C&& c, C0&& c0, Cn&&...cn)
#endif
      : DynamicDescriptor(std::forward<decltype(c)>(c))
    {
      extend(std::forward<decltype(c0)>(c0), std::forward<decltype(cn)>(cn)...);
    }


    /**
     * \brief Extend the DynamicDescriptor by appending to the end of the list.
     * \tparam Cs One or more \ref vector_space_descriptor objects
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      ((fixed_vector_space_descriptor<Cs> or euclidean_vector_space_descriptor<Cs> or std::same_as<std::decay_t<Cs>, DynamicDescriptor>) and ...)
#else
    template<typename...Cs, std::enable_if_t<(vector_space_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      ((fixed_vector_space_descriptor<Cs> or euclidean_vector_space_descriptor<Cs> or std::is_same_v<std::decay_t<Cs>, DynamicDescriptor>) and ...), int> = 0>
#endif
    DynamicDescriptor& extend(Cs&&...cs)
    {
      if (auto N = (0 + ... + get_vector_space_descriptor_component_count_of(cs)); N > 1)
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
     * \return Iterator marking the beginning of a vector containing a set of DynamicTypedVectorSpaceDescriptor objects.
     */
    auto begin() const { return dynamic_types.begin(); }


    /**
     * \return Iterator marking the end of a vector containing a set of DynamicTypedVectorSpaceDescriptor objects.
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
    void fill_tables_fixed(std::size_t i, std::size_t i_e, std::size_t t, const FixedDescriptor<Cs...>&)
    {
      (dynamic_types.emplace_back(AtomicType {Cs{}}), ...);
      extend_table_fixed<false, 0, Cs...>(i, t, i_e);
      extend_table_fixed<true, 0, Cs...>(i_e, t, i);
    }


    template<typename C, typename...Cs>
    void fill_tables_dynamic(std::size_t i, std::size_t i_e, std::size_t t, C&& c, Cs&&...cs)
    {
      if constexpr (euclidean_vector_space_descriptor<C>)
      {
        // \todo Add untyped \ref vector_space_descriptor without breaking them up.
        auto N = t + get_dimension_size_of(c);
        for (; t < N; ++i, ++i_e, ++t)
        {
          dynamic_types.emplace_back(AtomicType {Axis{}});
          index_table.emplace_back(t, 0, i_e);
          euclidean_index_table.emplace_back(t, 0, i);
        }
        fill_tables(i, i_e, t, std::forward<Cs>(cs)...);
      }
      else // C is DynamicDescriptor<T>.
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
      if constexpr (fixed_vector_space_descriptor<C>)
      {
        using red_C = canonical_fixed_vector_space_descriptor_t<std::decay_t<C>>;
        fill_tables_fixed(i, i_e, t, red_C {});
        fill_tables(i + dimension_size_of_v<C>, i_e + euclidean_dimension_size_of_v<C>,
          t + vector_space_descriptor_components_of<red_C>::value, std::forward<Cs>(cs)...);
      }
      else // dynamic_vector_space_descriptor<C>
      {
        fill_tables_dynamic(i, i_e, t, std::forward<C>(c), std::forward<Cs>(cs)...);
      }
    }


    void fill_tables(std::size_t, std::size_t, std::size_t) {}


    // ---------- comparison ---------- //

    template<typename It, typename EndIt, std::size_t N>
    static constexpr bool partial_compare(It it, EndIt endit, const Dimensions<N>&, FixedDescriptor<>)
    {
      return it == endit or it->is_euclidean();
    }

    template<typename It, typename EndIt, std::size_t N, typename C, typename...Cs>
    static constexpr bool partial_compare(It it, EndIt endit, const Dimensions<N>& d, FixedDescriptor<C, Cs...>)
    {
      if (it->is_euclidean())
      {
        std::size_t it_size = it->size();
        std::size_t d_size = get_dimension_size_of(d);
        if (it_size == d_size)
          return partial_compare(++it, endit, FixedDescriptor<C, Cs...> {});
        else if (it_size < d_size)
          return partial_compare(++it, endit, Dimensions {static_cast<std::size_t>(d_size - it_size)}, FixedDescriptor<C, Cs...> {});
        else // it_size > d_size
        {
          if constexpr (euclidean_vector_space_descriptor<C>)
            return partial_compare(it, endit, Dimensions {d_size + dimension_size_of_v<C>}, FixedDescriptor<Cs...> {});
          else
            return false;
        }
      }
      else return false;
    }

    template<typename It, typename EndIt>
    static constexpr bool partial_compare(It, EndIt, FixedDescriptor<>) { return true; }

    template<typename It, typename EndIt, typename C, typename...Cs>
    static constexpr bool partial_compare(It it, EndIt endit, FixedDescriptor<C, Cs...>)
    {
      if (it == endit) return true;
      else
      {
        if constexpr (euclidean_vector_space_descriptor<C>)
        {
          return partial_compare(it, endit, Dimensions {dimension_size_of_v<C>}, FixedDescriptor<Cs...> {});
        }
        else
        {
          if (it->get_type_index() == std::type_index {typeid(C)}) return partial_compare(++it, endit, FixedDescriptor<Cs...> {});
          else return false;
        }
      }
    }

  public:

    /**
     * \brief True if <code>this</code> is a subset or superset of the \ref vector_space_descriptor argument
     */
#ifdef __cpp_concepts
    template<typename Arg> requires fixed_vector_space_descriptor<Arg> or euclidean_vector_space_descriptor<Arg>
#else
    template<typename Arg, std::enable_if_t<fixed_vector_space_descriptor<Arg> or euclidean_vector_space_descriptor<Arg>, int> = 0>
#endif
    bool partially_matches(const Arg& arg) const
    {
      if constexpr (fixed_vector_space_descriptor<Arg>)
        return partial_compare(dynamic_types.begin(), dynamic_types.end(), canonical_fixed_vector_space_descriptor_t<Arg> {});
      else
        return partial_compare(dynamic_types.begin(), dynamic_types.end(), Dimensions<dynamic_size>(get_dimension_size_of(arg)), FixedDescriptor<> {});
    }


    /**
     * \overload
     */
    template<typename...S>
    bool partially_matches(const DynamicDescriptor<S...>& arg) const
    {
      // \todo Do a more sophisticated comparison.
      auto i = begin();
      for (auto j = arg.begin(); i != end() and j != arg.end(); ++i, ++j)
      {
        if (i->get_type_index() != j->get_type_index()) return false;
      }
      return true;
    }


    // ---------- operator+= is the same as extend ---------- //

  #ifdef __cpp_concepts
    template<vector_space_descriptor B>
  #else
    template<typename B>
  #endif
    constexpr decltype(auto) operator+=(B&& b) { return extend(std::forward<B>(b)); }


    // ---------- friends ---------- //

  friend struct interface::dynamic_vector_space_descriptor_traits<DynamicDescriptor>;


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
  DynamicDescriptor(Cs&&...) -> DynamicDescriptor<>;

  /**
   * \brief Deduce scalar type when the constructor's first argument is a DynamicDescriptor.
   */
  template<typename...S, typename C, typename...Cs>
  DynamicDescriptor(DynamicDescriptor<S...>&&, C&&, Cs&&...) -> DynamicDescriptor<S...>;

  /// \overload
  template<typename...S, typename C, typename...Cs>
  DynamicDescriptor(const DynamicDescriptor<S...>&, C&&, Cs&&...) -> DynamicDescriptor<S...>;


  namespace detail
  {
    template<typename Scalar, typename>
    struct is_allowable_scalar : std::false_type {};

    template<typename Scalar, typename...S>
    struct is_allowable_scalar<Scalar, internal::AnyAtomicVectorSpaceDescriptor<S...>>
      : std::bool_constant<(std::is_same_v<std::decay_t<Scalar>, S> or ...)> {};


    /**
     * \internal
     * \brief Tests whether Scalar is an allowable scalar type.
     * \tparam Scalar A \ref scalar_type
     * \tparam AtomicType An object of type internal::AnyAtomicVectorSpaceDescriptor
     */
#ifdef __cpp_concepts
    template<typename Scalar, typename AtomicType>
    concept allowable_scalar =
#else
    template<typename Scalar, typename AtomicType>
    constexpr bool allowable_scalar =
#endif
      scalar_type<Scalar> and is_allowable_scalar<Scalar, std::decay_t<AtomicType>>::value;

  } // namespace detail


  // --------- //
  //   traits  //
  // --------- //

  namespace interface
  {
    /**
     * \internal
     * \brief traits for DynamicDescriptor.
     */
    template<typename...AllowableScalarTypes>
    struct dynamic_vector_space_descriptor_traits<DynamicDescriptor<AllowableScalarTypes...>>
    {
    private:

      using AtomicType = typename DynamicDescriptor<AllowableScalarTypes...>::AtomicType;

    public:

      explicit constexpr dynamic_vector_space_descriptor_traits(const DynamicDescriptor<AllowableScalarTypes...>& t)
        : m_vector_space_descriptor {t} {};

      constexpr std::size_t get_size() const { return m_vector_space_descriptor.index_table.size(); }

      constexpr std::size_t get_euclidean_size() const { return m_vector_space_descriptor.euclidean_index_table.size(); }

      constexpr std::size_t get_component_count() const { return m_vector_space_descriptor.dynamic_types.size(); }

      constexpr bool is_euclidean() const
      {
        for (auto i = m_vector_space_descriptor.dynamic_types.begin(); i != m_vector_space_descriptor.dynamic_types.end(); ++i)
          if (not i->is_euclidean()) return false;
        return true;
      }

      static constexpr bool always_euclidean = false;

      static constexpr bool operations_defined = true;

  #ifdef __cpp_concepts
      scalar_type auto
      to_euclidean_element(const auto& g, std::size_t euclidean_local_index, std::size_t start) const
      requires requires (std::size_t i){ {g(i)} -> detail::allowable_scalar<AtomicType>; }
  #else
      template<typename G, std::enable_if_t<
        detail::allowable_scalar<typename std::invoke_result<const G&, std::size_t>::type, AtomicType>, int> = 0>
      auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start) const
  #endif
      {
        auto [tp, comp_euclidean_local_index, comp_start] = m_vector_space_descriptor.euclidean_index_table[euclidean_local_index];
        return m_vector_space_descriptor.dynamic_types[tp].to_euclidean_element(g, comp_euclidean_local_index, start + comp_start);
      }


      /**
       * \brief Maps a coordinate in Euclidean space to an element.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index relative to the original coordinates (starting at 0)
       * \param start The starting index within the Euclidean-transformed indices
       */
  #ifdef __cpp_concepts
      scalar_type auto
      from_euclidean_element(const auto& g, std::size_t local_index, std::size_t euclidean_start) const
      requires requires (std::size_t i){ {g(i)} -> detail::allowable_scalar<AtomicType>; }
  #else
      template<typename G, std::enable_if_t<
        detail::allowable_scalar<typename std::invoke_result<const G&, std::size_t>::type, AtomicType>, int> = 0>
      auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start) const
  #endif
      {
        auto [tp, comp_local_index, comp_euclidean_start] = m_vector_space_descriptor.index_table[local_index];
        return m_vector_space_descriptor.dynamic_types[tp].from_euclidean_element(g, comp_local_index, euclidean_start + comp_euclidean_start);
      }


      /**
       * \brief Perform modular wrapping of an element.
       * \details The wrapping operation is equivalent to mapping to, and then back from, Euclidean space.
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param local_index A local index relative to the original coordinates (starting at 0)
       * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
       */
  #ifdef __cpp_concepts
        scalar_type auto
        get_wrapped_component(const auto& g, std::size_t local_index, std::size_t start) const
        requires requires (std::size_t i){ {g(i)} -> detail::allowable_scalar<AtomicType>; }
  #else
        template<typename G, std::enable_if_t<
          detail::allowable_scalar<typename std::invoke_result<const G&, std::size_t>::type, AtomicType>, int> = 0>
        auto get_wrapped_component(const G& g, std::size_t local_index, std::size_t start) const
  #endif
      {
        auto [tp, comp_local_index, comp_start] = m_vector_space_descriptor.index_table[local_index];
        return m_vector_space_descriptor.dynamic_types[tp].get_wrapped_component(g, comp_local_index, start + local_index - comp_local_index);
      }


      /**
       * \brief Set an element and then perform any necessary modular wrapping.
       * \details The operation is equivalent to setting the angle and then mapping to, and then back from, Euclidean space.
       * \param s An element setter (<code>std::function&lt;void(std::size_t, Scalar)&rt;</code>)
       * \param g An element getter (<code>std::function&lt;Scalar(std::size_t)&rt;</code>)
       * \param x The scalar value to be set
       * \param local_index A local index relative to the original coordinates (starting at 0)
       * \param start The starting location of the angle within any larger set of \ref vector_space_descriptor
       */
  #ifdef __cpp_concepts
      void set_wrapped_component(const auto& s, const auto& g,
        const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start) const
      requires requires (std::size_t i){ s(x, i); {x} -> detail::allowable_scalar<AtomicType>; }
  #else
      template<typename S, typename G, std::enable_if_t<
        std::is_invocable<S, typename std::invoke_result<G, std::size_t>::type, std::size_t>::value and
        detail::allowable_scalar<typename std::invoke_result<G, std::size_t>::type, AtomicType>, int> = 0>
      void set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
        std::size_t local_index, std::size_t start) const
  #endif
      {
        auto [tp, comp_local_index, comp_start] = m_vector_space_descriptor.index_table[local_index];
        return m_vector_space_descriptor.dynamic_types[tp].set_wrapped_component(s, g, x, comp_local_index, start + local_index - comp_local_index);
      }

    private:

      const DynamicDescriptor<AllowableScalarTypes...>& m_vector_space_descriptor;

    };


  } // namespace interface

} // namespace OpenKalman


#endif //OPENKALMAN_DYNAMICDESCRIPTOR_HPP

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
#include <functional>
#include <typeindex>
#include "linear-algebra/values/values.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_component_count_of.hpp"

#include "StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/details/AnyAtomicVectorSpaceDescriptor.hpp"

namespace OpenKalman::descriptor
{
#ifdef __cpp_concepts
  template<value::number Scalar>
#else
  template<typename Scalar>
#endif
  struct DynamicDescriptor
  {
  private:
#ifndef __cpp_concepts
    static_assert(value::number<Scalar>);
#endif

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
      (sizeof...(Cs) != 1 or (not std::is_base_of_v<DynamicDescriptor, Cs> and ...)) and
      (sizeof...(Cs) == 1 or ((static_vector_space_descriptor<Cs> or dynamic_vector_space_descriptor<Cs>) and ...))
#else
    template<typename...Cs, std::enable_if_t<(vector_space_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_base_of_v<DynamicDescriptor, Cs> and ...)) and
      (sizeof...(Cs) == 1 or ((static_vector_space_descriptor<Cs> or dynamic_vector_space_descriptor<Cs>) and ...)),
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
     * \details This forwards the first argument to the move or forward constructor and then extends by adding the other arguments.
     * \param c A DynamicDescriptor object
     * \param cn A first \ref vector_space_descriptor.
     * \param cs A list of \ref vector_space_descriptor objects.
     */
#ifdef __cpp_concepts
    template<typename C, vector_space_descriptor C0, vector_space_descriptor...Cn> requires
      std::derived_from<std::decay_t<C>, DynamicDescriptor>
#else
    template<typename C, typename C0, typename...Cn, std::enable_if_t<
      std::is_base_of_v<DynamicDescriptor, std::decay_t<C>> and (vector_space_descriptor<Cn> and ...), int> = 0>
#endif
    DynamicDescriptor(C&& c, C0&& c0, Cn&&...cn) : DynamicDescriptor(std::forward<C>(c))
    {
      extend(std::forward<C0>(c0), std::forward<Cn>(cn)...);
    }


    /**
     * \brief Assign from another \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor D> requires (not std::is_base_of_v<DynamicDescriptor, D>)
#else
    template<typename D, std::enable_if_t<vector_space_descriptor<D> and (not std::is_base_of_v<DynamicDescriptor, D>), int> = 0>
#endif
    constexpr DynamicDescriptor& operator=(const D& d)
    {
      dynamic_types.clear();
      index_table.clear();
      euclidean_index_table.clear();
      dynamic_types.reserve(get_vector_space_descriptor_component_count_of(d));
      index_table.reserve(get_dimension_size_of(d));
      euclidean_index_table.reserve(get_euclidean_dimension_size_of(d));
      fill_tables(0, 0, 0, d);
      return *this;
    }


    /**
     * \brief Extend the DynamicDescriptor by appending to the end of the list.
     * \tparam Cs One or more \ref vector_space_descriptor objects
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      ((static_vector_space_descriptor<Cs> or euclidean_vector_space_descriptor<Cs> or std::same_as<std::decay_t<Cs>, DynamicDescriptor>) and ...)
#else
    template<typename...Cs, std::enable_if_t<(vector_space_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      ((static_vector_space_descriptor<Cs> or euclidean_vector_space_descriptor<Cs> or std::is_same_v<std::decay_t<Cs>, DynamicDescriptor>) and ...), int> = 0>
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
    void fill_tables_fixed(std::size_t i, std::size_t i_e, std::size_t t, const StaticDescriptor<Cs...>&)
    {
      (dynamic_types.emplace_back(detail::AnyAtomicVectorSpaceDescriptor<Scalar> {Cs{}}), ...);
      extend_table_fixed<false, 0, Cs...>(i, t, i_e);
      extend_table_fixed<true, 0, Cs...>(i_e, t, i);
    }


    template<typename C, typename...Cs>
    void fill_tables_dynamic(std::size_t i, std::size_t i_e, std::size_t t, C&& c, Cs&&...cs)
    {
      if constexpr (euclidean_vector_space_descriptor<C>)
      {
        auto N = t + get_dimension_size_of(c);
        for (; t < N; ++i, ++i_e, ++t)
        {
          dynamic_types.emplace_back(detail::AnyAtomicVectorSpaceDescriptor<Scalar> {Dimensions<1>{}});
          index_table.emplace_back(t, 0, i_e);
          euclidean_index_table.emplace_back(t, 0, i);
        }
        fill_tables(i, i_e, t, std::forward<Cs>(cs)...);
      }
      else // C is DynamicDescriptor.
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
      if constexpr (static_vector_space_descriptor<C>)
      {
        using red_C = internal::static_canonical_form_t<std::decay_t<C>>;
        fill_tables_fixed(i, i_e, t, red_C {});
        fill_tables(i + dimension_size_of_v<C>, i_e + euclidean_dimension_size_of_v<C>,
          t + vector_space_component_count<red_C>::value, std::forward<Cs>(cs)...);
      }
      else // dynamic_vector_space_descriptor<C>
      {
        fill_tables_dynamic(i, i_e, t, std::forward<C>(c), std::forward<Cs>(cs)...);
      }
    }


    void fill_tables(std::size_t, std::size_t, std::size_t) {}


    // ---------- comparison ---------- //

    template<typename It, typename EndIt, std::size_t N>
    static constexpr bool partial_compare(It it, EndIt endit, const Dimensions<N>&, StaticDescriptor<>)
    {
      return it == endit or it->is_euclidean();
    }


    template<typename It, typename EndIt, std::size_t N, typename C, typename...Cs>
    static constexpr bool partial_compare(It it, EndIt endit, const Dimensions<N>& d, StaticDescriptor<C, Cs...>)
    {
      if (it->is_euclidean())
      {
        std::size_t it_size = it->size();
        std::size_t d_size = get_dimension_size_of(d);
        if (it_size == d_size)
          return partial_compare(++it, endit, StaticDescriptor<C, Cs...> {});
        else if (it_size < d_size)
          return partial_compare(++it, endit, Dimensions {static_cast<std::size_t>(d_size - it_size)}, StaticDescriptor<C, Cs...> {});
        else // it_size > d_size
        {
          if constexpr (euclidean_vector_space_descriptor<C>)
            return partial_compare(it, endit, Dimensions {d_size + dimension_size_of_v<C>}, StaticDescriptor<Cs...> {});
          else
            return false;
        }
      }
      else return false;
    }


    template<typename It, typename EndIt>
    static constexpr bool partial_compare(It, EndIt, StaticDescriptor<>) { return true; }


    template<typename It, typename EndIt, typename C, typename...Cs>
    static constexpr bool partial_compare(It it, EndIt endit, StaticDescriptor<C, Cs...>)
    {
      if (it == endit) return true;
      else
      {
        if constexpr (euclidean_vector_space_descriptor<C>)
        {
          return partial_compare(it, endit, Dimensions {dimension_size_of_v<C>}, StaticDescriptor<Cs...> {});
        }
        else
        {
          if (it->get_type_index() == std::type_index {typeid(C)}) return partial_compare(++it, endit, StaticDescriptor<Cs...> {});
          else return false;
        }
      }
    }


    /**
     * \brief True if <code>this</code> is a subset or superset of the \ref vector_space_descriptor argument
     */
#ifdef __cpp_concepts
    template<typename Arg> requires static_vector_space_descriptor<Arg> or euclidean_vector_space_descriptor<Arg>
#else
    template<typename Arg, std::enable_if_t<static_vector_space_descriptor<Arg> or euclidean_vector_space_descriptor<Arg>, int> = 0>
#endif
    bool partially_matches(const Arg& arg) const
    {
      if constexpr (static_vector_space_descriptor<Arg>)
        return partial_compare(dynamic_types.begin(), dynamic_types.end(), internal::static_canonical_form_t<Arg> {});
      else
        return partial_compare(dynamic_types.begin(), dynamic_types.end(), Dimensions<dynamic_size> {get_dimension_size_of(arg)}, StaticDescriptor<> {});
    }


    /**
     * \overload
     */
    template<typename...S>
    bool partially_matches(const DynamicDescriptor<S...>& arg) const
    {
      auto i = begin();
      for (auto j = arg.begin(); i != end() and j != arg.end(); ++i, ++j)
      {
        if (i->get_type_index() != j->get_type_index()) return false;
      }
      return true;
    }


    template<typename A, typename B>
    static bool partial_match(const A& a, const B& b)
    {
      if constexpr (std::is_same_v<A, DynamicDescriptor>) return a.partially_matches(b);
      else return b.partially_matches(a);
    }

  public:

#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
    /**
     * \brief Comparison operator
     */
    template<vector_space_descriptor B>
    friend constexpr auto operator<=>(const DynamicDescriptor& a, const B& b)
    {
      if (partial_match(a, b))
        return std::partial_ordering {static_cast<std::size_t>(get_dimension_size_of(a)) <=> static_cast<std::size_t>(get_dimension_size_of(b))};
      else
        return std::partial_ordering::unordered;
    }


    /**
     * \brief Equality operator
     */
    template<vector_space_descriptor B>
    friend constexpr bool operator==(const DynamicDescriptor& a, const B& b)
    {
      return std::is_eq(a <=> b);
    }
#else
    template<typename T>
    struct is_DynamicDescriptor : std::false_type {};

    template<typename S>
    struct is_DynamicDescriptor<DynamicDescriptor<S>> : std::true_type {};


    friend constexpr bool operator==(const DynamicDescriptor& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) == static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename B, std::enable_if_t<vector_space_descriptor<B>, int> = 0>
    friend constexpr bool operator==(const DynamicDescriptor& a, const B& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) == static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename A, std::enable_if_t<vector_space_descriptor<A> and not is_DynamicDescriptor<A>::value, int> = 0>
    friend constexpr bool operator==(const A& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) == static_cast<std::size_t>(get_dimension_size_of(b));
    }


    friend constexpr bool operator<(const DynamicDescriptor& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) < static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename B, std::enable_if_t<vector_space_descriptor<B>, int> = 0>
    friend constexpr bool operator<(const DynamicDescriptor& a, const B& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) < static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename A, std::enable_if_t<vector_space_descriptor<A> and not is_DynamicDescriptor<A>::value, int> = 0>
    friend constexpr bool operator<(const A& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) < static_cast<std::size_t>(get_dimension_size_of(b));
    }


    friend constexpr bool operator>(const DynamicDescriptor& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) > static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename B, std::enable_if_t<vector_space_descriptor<B>, int> = 0>
    friend constexpr bool operator>(const DynamicDescriptor& a, const B& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) > static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename A, std::enable_if_t<vector_space_descriptor<A> and not is_DynamicDescriptor<A>::value, int> = 0>
    friend constexpr bool operator>(const A& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) > static_cast<std::size_t>(get_dimension_size_of(b));
    }


    friend constexpr bool operator!=(const DynamicDescriptor& a, const DynamicDescriptor& b)
    {
      return not partial_match(a, b) or static_cast<std::size_t>(get_dimension_size_of(a)) != static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename B, std::enable_if_t<vector_space_descriptor<B>, int> = 0>
    friend constexpr bool operator!=(const DynamicDescriptor& a, const B& b)
    {
      return not partial_match(a, b) or static_cast<std::size_t>(get_dimension_size_of(a)) != static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename A, std::enable_if_t<vector_space_descriptor<A> and not is_DynamicDescriptor<A>::value, int> = 0>
    friend constexpr bool operator!=(const A& a, const DynamicDescriptor& b)
    {
      return not partial_match(a, b) or static_cast<std::size_t>(get_dimension_size_of(a)) != static_cast<std::size_t>(get_dimension_size_of(b));
    }


    friend constexpr bool operator<=(const DynamicDescriptor& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) <= static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename B, std::enable_if_t<vector_space_descriptor<B>, int> = 0>
    friend constexpr bool operator<=(const DynamicDescriptor& a, const B& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) <= static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename A, std::enable_if_t<vector_space_descriptor<A> and not is_DynamicDescriptor<A>::value, int> = 0>
    friend constexpr bool operator<=(const A& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) <= static_cast<std::size_t>(get_dimension_size_of(b));
    }


    friend constexpr bool operator>=(const DynamicDescriptor& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) >= static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename B, std::enable_if_t<vector_space_descriptor<B>, int> = 0>
    friend constexpr bool operator>=(const DynamicDescriptor& a, const B& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) >= static_cast<std::size_t>(get_dimension_size_of(b));
    }


    template<typename A, std::enable_if_t<vector_space_descriptor<A> and not is_DynamicDescriptor<A>::value, int> = 0>
    friend constexpr bool operator>=(const A& a, const DynamicDescriptor& b)
    {
      return partial_match(a, b) and static_cast<std::size_t>(get_dimension_size_of(a)) >= static_cast<std::size_t>(get_dimension_size_of(b));
    }
#endif

  private:

    template<std::size_t n, std::size_t ni, std::size_t ne, typename It, typename EndIt>
    constexpr auto subtract(It, EndIt, StaticDescriptor<>) const
    {
      DynamicDescriptor ret;
      ret.dynamic_types.assign(dynamic_types.begin(), dynamic_types.end() - n);
      ret.index_table.assign(index_table.begin(), index_table.end() - ni);
      ret.euclidean_index_table.assign(euclidean_index_table.begin(), euclidean_index_table.end() - ne);
      return ret;
    }


    template<std::size_t n, std::size_t ni, std::size_t ne, typename It, typename EndIt, typename C, typename...Cs>
    constexpr auto subtract(It it, EndIt endit, StaticDescriptor<C, Cs...>) const
    {
      if (it == endit or (--endit)->get_type_index() != std::type_index {typeid(C)})
        throw std::invalid_argument {"Subtraction of incompatible vector_space_descriptor values"};
      else
        return subtract<n + 1, ni + dimension_size_of_v<C>, ne + euclidean_dimension_size_of_v<C>>(it, endit, StaticDescriptor<Cs...>{});
    }

  public:

    /**
     * \brief Minus operator when the operand is a \ref static_vector_space_descriptor
     */
#ifdef __cpp_concepts
    template<static_vector_space_descriptor B>
#else
    template<typename B, std::enable_if_t<static_vector_space_descriptor<B>, int> = 0>
#endif
    constexpr auto operator-(const B&) const
    {
      if constexpr (dimension_size_of_v<B> == 0) return *this;
      else
      {
        using F = static_reverse_t<internal::static_canonical_form_t<B>>;
        return subtract<0,0,0>(dynamic_types.begin(), dynamic_types.end(), F{});
      }
    }


    /**
     * \overload
     * \brief Minus operator when the operand is a dynamic \ref euclidean_vector_space_descriptor
     */
#ifdef __cpp_concepts
    template<euclidean_vector_space_descriptor B> requires dynamic_vector_space_descriptor<B>
#else
    template<typename B, std::enable_if_t<euclidean_vector_space_descriptor<B> and dynamic_vector_space_descriptor<B>, int> = 0>
#endif
    constexpr auto operator-(const B& b) const
    {
      // Compare the tails of a and b, from right to left
      auto i = dynamic_types.end();
      std::size_t j = 0;
      for (; j < get_dimension_size_of(b); ++j)
      {
        if (i == dynamic_types.begin() or not (--i)->is_euclidean())
          throw std::invalid_argument {"Subtraction of incompatible dynamic_vector_space_descriptor values"};
      }
      // Construct a dynamic vector space descriptor from the remainder of a
      DynamicDescriptor ret;
      ret.dynamic_types.assign(dynamic_types.begin(), i);
      ret.index_table.assign(index_table.begin(), index_table.end() - j);
      ret.euclidean_index_table.assign(euclidean_index_table.begin(), euclidean_index_table.end() - j);
      return ret;
    }


    /**
     * \overload
     * \brief Minus operator when the operand is another \ref DynamicDescriptor
     */
    template<typename S>
    friend constexpr auto operator-(const DynamicDescriptor& a, const DynamicDescriptor<S>& b)
    {
      // Compare the tails of a and b, from right to left
      auto i = a.dynamic_types.end();
      auto ii = a.index_table.end();
      auto ie = a.euclidean_index_table.end();
      for (auto j = b.dynamic_types.end(); j != b.dynamic_types.begin(); )
      {
        if (i == a.begin() or (--i)->get_type_index() != (--j)->get_type_index())
          throw std::invalid_argument {"Subtraction of incompatible dynamic_vector_space_descriptor values"};
        ii -= j->size();
        ie -= j->euclidean_size();
      }
      // Construct a dynamic vector space descriptor from the remainder of a
      DynamicDescriptor ret;
      ret.dynamic_types.assign(a.dynamic_types.begin(), i);
      ret.index_table.assign(a.index_table.begin(), ii);
      ret.euclidean_index_table.assign(a.euclidean_index_table.begin(), ie);
      return ret;
    }


    // ---------- operator+= is the same as extend ---------- //

#ifdef __cpp_concepts
    template<vector_space_descriptor B>
#else
    template<typename B>
#endif
    constexpr decltype(auto) operator+=(B&& b) { return extend(std::forward<B>(b)); }


    /**
     * \brief Split the object into head and tail parts.
     */
#ifdef __cpp_concepts
  template<value::index Offset, value::index Extent> requires
    (value::dynamic<Offset> or Offset::value >= 0) and (value::dynamic<Extent> or Extent::value >= 0)
#else
  template<typename Offset, typename Extent>
#endif
    constexpr auto
    slice(const Offset& offset, const Extent& extent) const
    {
      //std::cout << "size: " << std::size(index_table) << "; offset: " << offset << "; extent: " << extent << std::endl;
      //for (const auto& [i, local, e_start] : index_table)
      //  std::cout << "(" << i << ", " << local << ", " << e_start << ")" << std::endl;

      DynamicDescriptor ret;

      auto i_first = index_table.begin() + offset;
      auto i_last = i_first + extent;

      if (i_first != index_table.end())
      {
        if (std::get<1>(*i_first) != 0)
          throw std::invalid_argument {"DynamicDescriptor slice: offset does not coincide with a sub-descriptor boundary"};

        if (extent > 0)
        {
          if (i_last != index_table.end() and std::get<1>(*i_last) != 0)
            throw std::invalid_argument {"DynamicDescriptor slice: offset + extent does not coincide with a sub-descriptor boundary"};

          std::size_t t_first_val = std::get<0>(*i_first);
          auto t_first = dynamic_types.begin() + t_first_val;
          auto t_last = i_last == index_table.end() ? dynamic_types.end() : dynamic_types.begin() + std::get<0>(*i_last);

          std::ptrdiff_t e_first_val = std::get<2>(*i_first);
          auto e_first = euclidean_index_table.begin() + e_first_val;
          auto e_last = i_last == index_table.end() ? euclidean_index_table.end() : euclidean_index_table.begin() + std::get<2>(*i_last);

          ret.dynamic_types.assign(t_first, t_last);

          ret.index_table.reserve(extent);
          for (auto i = i_first; i != i_last; i++)
          {
            ret.index_table.emplace_back(
              std::get<0>(*i) - t_first_val,
              std::get<1>(*i),
              std::get<2>(*i) - e_first_val);
          }

          ret.euclidean_index_table.reserve(std::distance(e_first, e_last));
          for (auto e = e_first; e != e_last; e++)
          {
            ret.euclidean_index_table.emplace_back(
              std::get<0>(*e) - t_first_val,
              std::get<1>(*e),
              std::get<2>(*e) - offset);
          }
        }
      }

      return ret;
    }


    // ---------- friends ---------- //

#ifdef __cpp_concepts
    template<value::number>
#else
    template<typename>
#endif
    friend struct DynamicDescriptor;


    template<typename T>
    friend struct interface::dynamic_vector_space_descriptor_traits;

  protected:

    // ---------- tables ---------- //

    std::vector<detail::AnyAtomicVectorSpaceDescriptor<Scalar>> dynamic_types;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> index_table;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> euclidean_index_table;

  };


  // -------------------------------------- //
  //            deduction guides            //
  // -------------------------------------- //

  /**
   * \brief Deduce scalar type when the constructor's first argument is a DynamicDescriptor.
   */
  template<typename S, typename Arg, typename...Args>
  DynamicDescriptor(DynamicDescriptor<S>&&, Arg&&, Args&&...) -> DynamicDescriptor<S>;

  /// \overload
  template<typename S, typename Arg, typename...Args>
  DynamicDescriptor(const DynamicDescriptor<S>&, Arg&&, Args&&...) -> DynamicDescriptor<S>;


} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for DynamicDescriptor.
   */
  template<typename Scalar>
  struct dynamic_vector_space_descriptor_traits<descriptor::DynamicDescriptor<Scalar>>
  {
  private:

    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

  public:

    explicit constexpr dynamic_vector_space_descriptor_traits(const descriptor::DynamicDescriptor<Scalar>& t)
      : m_vector_space_descriptor {t} {};


    [[nodiscard]] constexpr std::size_t get_size() const { return m_vector_space_descriptor.index_table.size(); }


    [[nodiscard]] constexpr std::size_t get_euclidean_size() const { return m_vector_space_descriptor.euclidean_index_table.size(); }


    [[nodiscard]] constexpr std::size_t get_component_count() const { return m_vector_space_descriptor.dynamic_types.size(); }


    [[nodiscard]] constexpr bool is_euclidean() const
    {
      for (auto i = m_vector_space_descriptor.dynamic_types.begin(); i != m_vector_space_descriptor.dynamic_types.end(); ++i)
        if (not i->is_euclidean()) return false;
      return true;
    }


    static constexpr bool always_euclidean = false;


#ifdef __cpp_concepts
    value::number auto
    to_euclidean_element(const std::convertible_to<Getter> auto& g, std::size_t euclidean_local_index, std::size_t start) const
#else
    template<typename G, std::enable_if_t<std::is_convertible_v<G, Getter>, int> = 0>
    auto to_euclidean_element(const G& g, std::size_t euclidean_local_index, std::size_t start) const
#endif
    {
      auto [tp, comp_euclidean_local_index, comp_start] = m_vector_space_descriptor.euclidean_index_table[euclidean_local_index];
      return m_vector_space_descriptor.dynamic_types[tp].to_euclidean_element(g, comp_euclidean_local_index, start + comp_start);
    }


#ifdef __cpp_concepts
    value::number auto
    from_euclidean_element(const std::convertible_to<Getter> auto& g, std::size_t local_index, std::size_t euclidean_start) const
#else
    template<typename G, std::enable_if_t<is_convertible_v<std::is_convertible_v<G, Getter>, int> = 0>
    auto from_euclidean_element(const G& g, std::size_t local_index, std::size_t euclidean_start) const
#endif
    {
      auto [tp, comp_local_index, comp_euclidean_start] = m_vector_space_descriptor.index_table[local_index];
      return m_vector_space_descriptor.dynamic_types[tp].from_euclidean_element(g, comp_local_index, euclidean_start + comp_euclidean_start);
    }


#ifdef __cpp_concepts
    value::number auto
    get_wrapped_component(const std::convertible_to<Getter> auto& g, std::size_t local_index, std::size_t start) const
#else
    template<typename G, std::enable_if_t<is_convertible_v<std::is_convertible_v<G, Getter>, int> = 0>
    auto get_wrapped_component(const G& g, std::size_t local_index, std::size_t start) const
#endif
    {
      auto [tp, comp_local_index, comp_start] = m_vector_space_descriptor.index_table[local_index];
      return m_vector_space_descriptor.dynamic_types[tp].get_wrapped_component(g, comp_local_index, start + local_index - comp_local_index);
    }


#ifdef __cpp_concepts
    void set_wrapped_component(const std::convertible_to<Setter> auto& s, const std::convertible_to<Getter> auto& g,
      const std::decay_t<std::invoke_result_t<decltype(g), std::size_t>>& x, std::size_t local_index, std::size_t start) const
#else
    template<typename S, typename G, std::enable_if_t<std::is_convertible_v<S, Setter> and std::is_convertible_v<G, Getter>, int> = 0>
    void set_wrapped_component(const S& s, const G& g, const std::decay_t<typename std::invoke_result<G, std::size_t>::type>& x,
      std::size_t local_index, std::size_t start) const
#endif
    {
      auto [tp, comp_local_index, comp_start] = m_vector_space_descriptor.index_table[local_index];
      return m_vector_space_descriptor.dynamic_types[tp].set_wrapped_component(s, g, x, comp_local_index, start + local_index - comp_local_index);
    }

  private:

    const descriptor::DynamicDescriptor<Scalar>& m_vector_space_descriptor;

  };

} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_DYNAMICDESCRIPTOR_HPP

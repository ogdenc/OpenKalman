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
#include <functional>
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_component_count_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"
#include "StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/index.hpp"
#include "internal/AnyAtomicVectorSpaceDescriptor.hpp"

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

    template<std::size_t ix = 0, typename C>
    constexpr void
    extend_table_fixed_t(const C& c)
    {
      if constexpr (ix < std::tuple_size_v<C>)
      {
        dynamic_types.emplace_back(std::get<ix>(c));
        extend_table_fixed_t<ix + 1>(c);
      }
    }


    template<std::size_t ix = 0, std::size_t local_index = 0, typename C>
    constexpr void
    extend_table_fixed_i(std::size_t i, std::size_t i_e, std::size_t t, const C& c)
    {
      if constexpr (ix < std::tuple_size_v<C>)
      {
        if constexpr (local_index < dimension_size_of_v<std::tuple_element_t<ix, C>>)
        {
          index_table.emplace_back(t, local_index, i_e);
          extend_table_fixed_i<ix, local_index + 1>(i + 1, i_e, t, c);
        }
        else
        {
          extend_table_fixed_i<ix + 1, 0>(i, i_e + euclidean_dimension_size_of_v<std::tuple_element_t<ix, C>>, t + 1, c);
        }
      }
    }


    template<std::size_t ix = 0, std::size_t e_local_index = 0, typename C>
    constexpr void
    extend_table_fixed_e(std::size_t i, std::size_t i_e, std::size_t t, const C& c)
    {
      if constexpr (ix < std::tuple_size_v<C>)
      {
        if constexpr (e_local_index < euclidean_dimension_size_of_v<std::tuple_element_t<ix, C>>)
        {
          euclidean_index_table.emplace_back(t, e_local_index, i);
          extend_table_fixed_e<ix, e_local_index + 1>(i, i_e + 1, t, c);
        }
        else
        {
          extend_table_fixed_e<ix + 1, 0>(i + dimension_size_of_v<std::tuple_element_t<ix, C>>, i_e, t + 1, c);
        }
      }
    }


    constexpr void
    fill_tables(std::size_t i, std::size_t e, std::size_t t) {}


    template<typename C, typename...Cs>
    constexpr void
    fill_tables(std::size_t i, std::size_t e, std::size_t t, C&& c, Cs&&...cs)
    {
      if constexpr (static_vector_space_descriptor<C>)
      {
        auto coll = get_collection_of(c);
        extend_table_fixed_t(coll);
        extend_table_fixed_i(i, e, t, coll);
        extend_table_fixed_e(i, e, t, coll);
        fill_tables(
          i + get_dimension_size_of(c),
          e + get_euclidean_dimension_size_of(c),
          t + get_vector_space_descriptor_component_count_of(c),
          std::forward<Cs>(cs)...);
      }
      else if constexpr (euclidean_vector_space_descriptor<C>)
      {
        auto N = t + get_dimension_size_of(c);
        for (; t < N; ++i, ++e, ++t)
        {
          dynamic_types.emplace_back(Dimensions<1>{});
          index_table.emplace_back(t, 0, e);
          euclidean_index_table.emplace_back(t, 0, i);
        }
        fill_tables(i, e, t, std::forward<Cs>(cs)...);
      }
      else // C is dynamic
      {
        auto new_i = i + c.index_table.size();
        auto new_e = e + c.euclidean_index_table.size();
        auto new_t = t + c.dynamic_types.size();

        for (const auto& j : c.dynamic_types)
          dynamic_types.emplace_back(j);

        for (const auto& j : c.index_table)
          index_table.emplace_back(t + std::get<0>(j), std::get<1>(j), e + std::get<2>(j));

        for (const auto& j : c.euclidean_index_table)
          euclidean_index_table.emplace_back(t + std::get<0>(j), std::get<1>(j), i + std::get<2>(j));

        fill_tables(new_i, new_e, new_t, std::forward<Cs>(cs)...);
      }
    }

  public:

    /**
     * \brief Default constructor.
     */
    constexpr DynamicDescriptor() = default;


    /**
     * \brief Constructor taking any number of \ref vector_space_descriptor Cs.
     * \details \ref vector_space_descriptor objects Cs can be either fixed or dynamic. If dynamic, the vector_space_descriptor must be either
     * untyped or the same type as this DynamicDescriptor.
     * \tparam Cs A list of sets of \ref vector_space_descriptor for a set of indices.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_base_of_v<DynamicDescriptor, Cs> and ...))
#else
    template<typename...Cs, std::enable_if_t<(vector_space_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_base_of_v<DynamicDescriptor, Cs> and ...)),
      int> = 0>
#endif
    explicit constexpr DynamicDescriptor(Cs&&...cs)
    {
      dynamic_types.reserve((0 + ... + get_vector_space_descriptor_component_count_of(cs)));
      index_table.reserve((0 + ... + get_dimension_size_of(cs)));
      euclidean_index_table.reserve((0 + ... + get_euclidean_dimension_size_of(cs)));
      fill_tables(0, 0, 0, std::forward<Cs>(cs)...);
    }


    /**
     * \brief Extend the DynamicDescriptor by appending to the end of the list.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor B>
#else
    template<typename B>
#endif
    constexpr decltype(auto)
    operator+=(B&& b)
    {
      if (auto N = get_vector_space_descriptor_component_count_of(b); N > 1)
      {
        if (dynamic_types.capacity() < std::size(dynamic_types) + N)
          dynamic_types.reserve(dynamic_types.capacity() * 2);

        if (index_table.capacity() < (std::size(index_table) + get_dimension_size_of(b)))
          index_table.reserve(index_table.capacity() * 2);

        if (euclidean_index_table.capacity() < (euclidean_index_table.size() + get_euclidean_dimension_size_of(b)))
          euclidean_index_table.reserve(euclidean_index_table.capacity() * 2);
      }
      fill_tables(std::size(index_table), std::size(euclidean_index_table), std::size(dynamic_types), std::forward<B>(b));
      return *this;
    }


    /**
     * \brief Assign from another \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor D> requires (not std::is_base_of_v<DynamicDescriptor, D>)
#else
    template<typename D, std::enable_if_t<vector_space_descriptor<D> and (not std::is_base_of_v<DynamicDescriptor, D>), int> = 0>
#endif
    constexpr DynamicDescriptor&
    operator=(const D& d)
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


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::vector_space_traits;

  protected:

    // ---------- tables ---------- //

    std::vector<internal::AnyAtomicVectorSpaceDescriptor<Scalar>> dynamic_types;
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
  struct vector_space_traits<descriptor::DynamicDescriptor<Scalar>>
  {
  private:

    using T = descriptor::DynamicDescriptor<Scalar>;
    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

  public:

    using scalar_type = Scalar;


    static constexpr auto
    size(const T& t) { return t.index_table.size(); }


    static constexpr auto
    euclidean_size(const T& t) { return t.euclidean_index_table.size(); }


    template<typename Arg>
    static constexpr decltype(auto)
    collection(Arg&& arg) { return std::forward<Arg>(arg).dynamic_types; }


    static constexpr auto
    is_euclidean(const T& t)
    {
      for (const auto& i : t.dynamic_types)
        if (not descriptor::get_vector_space_descriptor_is_euclidean(i)) return false;
      return true;
    }


    template<typename Arg>
    static constexpr auto
    subtract(const T& t, const Arg& arg)
    {
      return t.operator-(arg);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar> and value::index<L> and value::index<S>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index, const S& start)
#endif
    {
      auto [tp, comp_euclidean_local_index, comp_start] = t.euclidean_index_table[euclidean_local_index];
      return descriptor::to_euclidean_element(t.dynamic_types[tp], g, comp_euclidean_local_index, start + comp_start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index, const value::index auto& euclidean_start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index, const S& euclidean_start)
#endif
    {
      auto [tp, comp_local_index, comp_euclidean_start] = t.index_table[local_index];
      return descriptor::from_euclidean_element(t.dynamic_types[tp], g, comp_local_index, euclidean_start + comp_euclidean_start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index, const S& start)
#endif
    {
      auto [tp, comp_local_index, comp_start] = t.index_table[local_index];
      return descriptor::get_wrapped_component(t.dynamic_types[tp], g, comp_local_index, start + local_index - comp_local_index);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const Scalar& x,
      const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename L, typename S, std::enable_if_t<
      value::index<L> and value::index<S> and
      std::is_invocable<const Setter&, const Scalar&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const Scalar& x, const L& local_index, const S& start)
#endif
    {
      auto [tp, comp_local_index, comp_start] = t.index_table[local_index];
      descriptor::set_wrapped_component(t.dynamic_types[tp], s, g, x, comp_local_index, start + local_index - comp_local_index);
    }

  };

} // namespace OpenKalman::interface


#endif //OPENKALMAN_DYNAMICDESCRIPTOR_HPP

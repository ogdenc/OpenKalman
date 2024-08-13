/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definitions for CovarianceBase, case 4.
 */

#ifndef OPENKALMAN_COVARIANCEBASE4_HPP
#define OPENKALMAN_COVARIANCEBASE4_HPP

#include <utility>

namespace OpenKalman::internal
{
  // ======================================CASE 4======================================
  /*
   * \internal
   * \anchor CovarianceBaseCase4
   * \brief Base of Covariance and SquareRootCovariance classes (Case 4).
   * \details This specialization is operative if NestedMatrix is not self-contained and
   * # Derived is a square root or NestedMatrix is not self-adjoint; and
   * # Derived is not a square root or NestedMatrix is not triangular.
   * In this case, NestedMatrix and the cholesky nested matrix are different.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename NestedMatrix> requires
    (not case1or2<Derived, NestedMatrix>) and (not self_contained<NestedMatrix>)
  struct CovarianceBase<Derived, NestedMatrix>
#else
  template<typename Derived, typename NestedMatrix>
  struct CovarianceBase<Derived, NestedMatrix, std::enable_if_t<
    (not case1or2<Derived, NestedMatrix>) and (not self_contained<NestedMatrix>)>>
#endif
  : AdapterBase<Derived, NestedMatrix>
  {
  private:

    using Base = AdapterBase<Derived, NestedMatrix>;

    using Scalar = scalar_type_of_t<NestedMatrix>;


#ifdef __cpp_concepts
    template<typename, typename>
#else
    template<typename, typename, typename>
#endif
    friend struct CovarianceBase;

  protected:

    using CholeskyNestedMatrix = std::conditional_t<
      diagonal_matrix<NestedMatrix>,
      typename MatrixTraits<std::decay_t<NestedMatrix>>::template DiagonalMatrixFrom<>,
      std::conditional_t<triangular_matrix<NestedMatrix>,
        typename MatrixTraits<std::decay_t<NestedMatrix>>::template SelfAdjointMatrixFrom<>,
        typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularMatrixFrom<>>>;

  private:

    // Whether *this owns the value of cholesky_nested.
    const bool owns_cholesky_nested;


    // Storage for cholesky_nested. Only used if owns_cholesky_nested == true.
    CholeskyNestedMatrix cholesky_nested_store;


    // Link to the cholesky nested matrix, which is self-adjoint if Covariance, triangular if SquareRootCovariance.
    CholeskyNestedMatrix& cholesky_nested_link;

  protected:

    /**
     * \internal
     * \brief The cholesky nested matrix: self-adjoint if Covariance, triangular if SquareRootCovariance
     */
    auto& cholesky_nested_matrix() & { return cholesky_nested_link; }


    /// \overload
    const auto& cholesky_nested_matrix() const & { return cholesky_nested_link; }


    /// \overload
    const auto& cholesky_nested_matrix() &&
    {
      if (owns_cholesky_nested) return std::move(*this).cholesky_nested_store;
      else return cholesky_nested_link;
    }


    /// \overload
    const auto& cholesky_nested_matrix() const &&
    {
      if (owns_cholesky_nested) return std::move(*this).cholesky_nested_store;
      else return cholesky_nested_link;
    }


    /**
     * \internal
     * \brief The synchronization direction. 0 == synchronized, 1 = forward, -1 = reverse.
     */
    const std::function<int()> synchronization_direction;


    /**
     * \internal
     * \brief Synchronize nested_object to cholesky_nested_matrix.
     */
    const std::function<void()> synchronize_forward;


    /**
     * \internal
     * \brief Synchronize cholesky_nested_matrix to nested_object.
     */
    const std::function<void()> synchronize_reverse;


    /**
     * \internal
     * \brief Indicate that the nested matrix has changed.
     */
    const std::function<void()> mark_nested_matrix_changed;


    /**
     * \internal
     * \brief Indicate that the cholesky nested matrix has changed.
     */
    const std::function<void()> mark_cholesky_nested_matrix_changed;


    /**
     * \internal
     * \brief Indicate that the covariance is synchronized.
     */
    const std::function<void()> mark_synchronized;

  private:

    void maybe_synchronize_reverse()
    {
      if constexpr (modifiable<NestedMatrix, decltype(to_covariance_nestable<NestedMatrix>(cholesky_nested_store))>)
        Base::nested_object() = to_covariance_nestable<NestedMatrix>(cholesky_nested_store);
      else
        throw (std::logic_error("CovarianceBase4 maybe_synchronize_reverse: NestedMatrix is not modifiable"));
    }

  public:

    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other) = default;


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) = default;


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is self-contained and the same kind as its
     * cholesky nested matrix and the same as NestedMatrix. This only occurs when constructing the
     * square or square_root of a covariance.
     * In other words, Arg is a \ref CovarianceBaseCase1 "Case 1" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and self_contained<nested_object_of_t<Arg>> and
      (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      self_contained<nested_object_of_t<Arg>> and
      (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>),
      int> = 0>
#endif
    CovarianceBase(Arg& arg) noexcept
      : Base {arg.nested_object()},
        owns_cholesky_nested {true},
        cholesky_nested_store {},
        cholesky_nested_link {cholesky_nested_store},
        synchronization_direction {[]() -> int { return 1; }},
        synchronize_forward {[this] {
          cholesky_nested_store = to_covariance_nestable<CholeskyNestedMatrix>(Base::nested_object());
        }},
        synchronize_reverse {[this] { maybe_synchronize_reverse(); }},
        mark_nested_matrix_changed {[] {}},
        mark_cholesky_nested_matrix_changed {[] {}},
        mark_synchronized {[] {}}
    {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is not self-contained and the same kind as its
     * cholesky nested matrix and the same as NestedMatrix. This only occurs when constructing the
     * square or square_root of a covariance.
     * In other words, Arg is a \ref CovarianceBaseCase2 "Case 2" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and (not self_contained<nested_object_of_t<Arg>>) and
      (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      (not self_contained<nested_object_of_t<Arg>>) and
      (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>),
      int> = 0>
#endif
    CovarianceBase(Arg& arg) noexcept
      : Base {arg.nested_object()},
        owns_cholesky_nested {true},
        cholesky_nested_store {},
        cholesky_nested_link {cholesky_nested_store},
        synchronization_direction {[]() -> int { return 1; }},
        synchronize_forward {[this, &arg] {
          if (arg.synchronization_direction() < 0) arg.synchronize_reverse();
          cholesky_nested_store = arg.cholesky_nested_matrix();
        }},
        synchronize_reverse {[this] {
          maybe_synchronize_reverse();
          mark_nested_matrix_changed();
        }},
        mark_nested_matrix_changed {arg.mark_nested_matrix_changed},
        mark_cholesky_nested_matrix_changed {[] {}},
        mark_synchronized {[] {}}
    {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is not self-contained and the same kind as its
     * cholesky nested matrix and the same as NestedMatrix. This only occurs when constructing the
     * square or square_root of a covariance.
     * In other words, Arg is a \ref CovarianceBaseCase2 "Case 2" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires case1or2<Arg> and (not self_contained<nested_object_of_t<Arg>>) and
      (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>) and
      std::is_rvalue_reference_v<Arg&&>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and case1or2<Arg> and
      (not self_contained<nested_object_of_t<Arg>>) and
      (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>) and
      std::is_rvalue_reference_v<Arg&&>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base {std::move(arg).nested_object()},
        owns_cholesky_nested {true},
        cholesky_nested_store {},
        cholesky_nested_link {cholesky_nested_store},
        synchronization_direction {[]() -> int { return 1; }},
        synchronize_forward {
          [this, d = std::move(arg).synchronization_direction, sr = std::move(arg).synchronize_reverse] {
          if (d() < 0) sr();
          cholesky_nested_store = to_covariance_nestable<CholeskyNestedMatrix>(Base::nested_object());
        }},
        synchronize_reverse {[this] {
          maybe_synchronize_reverse();
          mark_nested_matrix_changed();
        }},
        mark_nested_matrix_changed {std::move(arg).mark_nested_matrix_changed},
        mark_cholesky_nested_matrix_changed {[] {}},
        mark_synchronized {[] {}}
    {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is self-contained and is of a different
     * type from its cholesky nested matrix.
     * In other words, Arg is a \ref CovarianceBaseCase3 "Case 3" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not case1or2<Arg>) and self_contained<nested_object_of_t<Arg>> and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not case1or2<Arg>) and self_contained<nested_object_of_t<Arg>> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg& arg) noexcept
      : Base {arg.nested_object()},
        owns_cholesky_nested {false},
        cholesky_nested_store {},
        cholesky_nested_link {arg.cholesky_nested},
        synchronization_direction {[&arg] { return arg.synch_direction; }},
        synchronize_forward {[&arg] { arg.synchronize_forward(); }},
        synchronize_reverse {[&arg] { arg.synchronize_reverse(); }},
        mark_nested_matrix_changed {[&arg] { arg.mark_nested_matrix_changed(); }},
        mark_cholesky_nested_matrix_changed {[&arg] { arg.mark_cholesky_nested_matrix_changed(); }},
        mark_synchronized {[&arg] { arg.mark_synchronized(); }}
    {}


    /**
     * \brief Construct from an another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is not self-contained and is of a different
     * type from its cholesky nested matrix, and Arg is an lvalue reference.
     * In other words, Arg is a \ref CovarianceBaseCase4 "Case 4" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not case1or2<Arg>) and (not self_contained<nested_object_of_t<Arg>>) and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not case1or2<Arg>) and (not self_contained<nested_object_of_t<Arg>>) and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg& arg) noexcept
      : Base {arg.nested_object()},
        owns_cholesky_nested {false},
        cholesky_nested_store {},
        cholesky_nested_link {arg.cholesky_nested_link},
        synchronization_direction {arg.synchronization_direction},
        synchronize_forward {arg.synchronize_forward},
        synchronize_reverse {arg.synchronize_reverse},
        mark_nested_matrix_changed {arg.mark_nested_matrix_changed},
        mark_cholesky_nested_matrix_changed {arg.mark_cholesky_nested_matrix_changed},
        mark_synchronized {arg.mark_synchronized}
    {}


    /**
     * \brief Construct from another \ref covariance.
     * \details This overload is operative if Arg's nested matrix is not self-contained and is of a different
     * type from its cholesky nested matrix, and Arg is an rvalue reference.
     * In other words, Arg is a \ref CovarianceBaseCase4 "Case 4" covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not case1or2<Arg>) and (not self_contained<nested_object_of_t<Arg>>) and
      (not std::derived_from<std::decay_t<Arg>, CovarianceBase>) and std::is_rvalue_reference_v<Arg&&>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not case1or2<Arg>) and (not self_contained<nested_object_of_t<Arg>>) and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>) and std::is_rvalue_reference_v<Arg&&>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base {std::move(arg).nested_object()},
        owns_cholesky_nested {arg.owns_cholesky_nested},
        cholesky_nested_store {std::move(arg).cholesky_nested_store},
        cholesky_nested_link {owns_cholesky_nested ? cholesky_nested_store : arg.cholesky_nested_link},
        synchronization_direction {owns_cholesky_nested ? []() -> int { return 1; } : std::move(arg).synchronization_direction},
        synchronize_forward {
          owns_cholesky_nested ?
          [this, d = std::move(arg).synchronization_direction, sr = std::move(arg).synchronize_reverse] {
            if (d() < 0) sr();
            cholesky_nested_store = to_covariance_nestable<CholeskyNestedMatrix>(Base::nested_object());
          } :
          std::move(arg).synchronize_forward
        },
        synchronize_reverse {
          owns_cholesky_nested ? std::function<void()> {[this] {
            maybe_synchronize_reverse();
            mark_nested_matrix_changed();
          }} : std::move(arg).synchronize_reverse},
        mark_nested_matrix_changed {std::move(arg).mark_nested_matrix_changed},
        mark_cholesky_nested_matrix_changed {std::move(arg).mark_cholesky_nested_matrix_changed},
        mark_synchronized {std::move(arg).mark_synchronized}
    {}


    /**
     * \brief Construct from a \ref covariance_nestable of matching triangle/self-adjoint type.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and
      (not diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and
      (triangular_matrix<Arg> == triangular_matrix<NestedMatrix>) and (not diagonal_matrix<Arg>), int> = 0>
#endif
    explicit CovarianceBase(Arg&& arg) noexcept
      : Base {std::forward<Arg>(arg)},
        owns_cholesky_nested {true},
        cholesky_nested_store {},
        cholesky_nested_link {cholesky_nested_store},
        synchronization_direction {[]() -> int { return 1; }},
        synchronize_forward {[this] {
          cholesky_nested_store = to_covariance_nestable<CholeskyNestedMatrix>(Base::nested_object());
        }},
        synchronize_reverse {[this] { maybe_synchronize_reverse(); }},
        mark_nested_matrix_changed {[] {}},
        mark_cholesky_nested_matrix_changed {[] {}},
        mark_synchronized {[] {}}
    {}


    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        if (other.synchronization_direction() > 0)
        {
          Base::nested_object() = other.nested_object();
          mark_nested_matrix_changed();
        }
        else if (other.synchronization_direction() < 0)
        {
          cholesky_nested_link = other.cholesky_nested_link;
          mark_cholesky_nested_matrix_changed();
          if (synchronization_direction() > 0) synchronize_reverse(); //< if synchronization_direction() is always 1
        }
        else // other.synchronization_direction() == 0
        {
          Base::nested_object() = other.nested_object();
          mark_synchronized();
          // Don't bother updating cholesky_nested() if synchronization_direction() is always 1:
          if (synchronization_direction() <= 0) cholesky_nested_link = other.cholesky_nested_link;
        }
      }
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        if (other.synchronization_direction() > 0)
        {
          Base::nested_object() = std::move(other).nested_object();
          mark_nested_matrix_changed();
        }
        else if (other.synchronization_direction() < 0)
        {
          cholesky_nested_link = std::move(other).cholesky_nested_matrix();
          mark_cholesky_nested_matrix_changed();
          if (synchronization_direction() > 0) synchronize_reverse(); //< if synchronization_direction() is always 1
        }
        else // other.synchronization_direction() == 0
        {
          Base::nested_object() = std::move(other).nested_object();
          mark_synchronized();
          // Don't bother updating cholesky_nested() if synchronization_direction() is always 1:
          if (synchronization_direction() <= 0) cholesky_nested_link = std::move(other).cholesky_nested_matrix();
        }
      }
      return *this;
    }


    /**
     * \internal
     * \brief Assign from another \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, CovarianceBase>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (not std::is_base_of_v<CovarianceBase, std::decay_t<Arg>>), int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr(not (zero<nested_object_of_t<Arg>> and zero<NestedMatrix>) and
        not (identity_matrix<nested_object_of_t<Arg>> and identity_matrix<NestedMatrix>))
      {
        if constexpr (triangular_matrix<nested_object_of_t<Arg>> == triangular_matrix<NestedMatrix> and
          not diagonal_matrix<Arg>)
        {
          if (arg.synchronization_direction() > 0)
          {
            Base::nested_object() = to_covariance_nestable<NestedMatrix>(OpenKalman::nested_object(std::forward<Arg>(arg)));
            mark_nested_matrix_changed();
          }
          else if (arg.synchronization_direction() < 0)
          {
            cholesky_nested_link = to_covariance_nestable<CholeskyNestedMatrix>(
              std::forward<Arg>(arg).cholesky_nested_matrix());
            mark_cholesky_nested_matrix_changed();
            if (synchronization_direction() > 0) synchronize_reverse();
          }
          else // arg.synchronization_direction() == 0
          {
            Base::nested_object() = to_covariance_nestable<NestedMatrix>(OpenKalman::nested_object(std::forward<Arg>(arg)));
            mark_synchronized();
            if (not case1or2<Arg> and synchronization_direction() <= 0)
            {
              cholesky_nested_link = to_covariance_nestable<CholeskyNestedMatrix>(
                std::forward<Arg>(arg).cholesky_nested_matrix());
            }
          }
        }
        else
        {
          if (arg.synchronization_direction() < 0)
          {
            Base::nested_object() = to_covariance_nestable<NestedMatrix>(
              std::forward<Arg>(arg).cholesky_nested_matrix());
            mark_nested_matrix_changed();
          }
          else if (case1or2<Arg> or arg.synchronization_direction() > 0)
          {
            cholesky_nested_link = to_covariance_nestable<CholeskyNestedMatrix>(OpenKalman::nested_object(std::forward<Arg>(arg)));
            mark_cholesky_nested_matrix_changed();
            if (synchronization_direction() > 0) synchronize_reverse();
          }
          else // arg.synchronization_direction() == 0 and (Case 3 or 4)
          {
            Base::nested_object() = to_covariance_nestable<NestedMatrix>(
              std::forward<Arg>(arg).cholesky_nested_matrix());
            mark_synchronized();
            if (synchronization_direction() <= 0)
            {
              cholesky_nested_link = to_covariance_nestable<CholeskyNestedMatrix>(OpenKalman::nested_object(std::forward<Arg>(arg)));
            }
          }
        }
      }
      return *this;
    }


    /**
     * \brief Assign from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr(not (zero<Arg> and zero<NestedMatrix>) and
        not (identity_matrix<Arg> and identity_matrix<NestedMatrix>))
      {
        if constexpr(triangular_matrix<Arg> == triangular_matrix<NestedMatrix> and not diagonal_matrix<Arg>)
        {
          Base::nested_object() = to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
          mark_nested_matrix_changed();
        }
        else
        {
          cholesky_nested_link = to_covariance_nestable<CholeskyNestedMatrix>(std::forward<Arg>(arg));
          mark_cholesky_nested_matrix_changed();
          if (synchronization_direction() > 0) synchronize_reverse();
        }
      }
      return *this;
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementAccessor object.
     */
    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr(writable_by_component<NestedMatrix, 2> and writable_by_component<CholeskyNestedMatrix, 2>)
      {
        return ElementAccessor(cholesky_nested_link, i, j,
          [this] { if (synchronization_direction() > 0) synchronize_forward(); },
          [this] {
            mark_cholesky_nested_matrix_changed();
            if (synchronization_direction() > 0) synchronize_reverse();
          });
      }
      else
      {
        return ElementAccessor(cholesky_nested_link, i, j,
          [this] { if (synchronization_direction() > 0) synchronize_forward(); });
      }
    }

    /// \overload
    auto operator() (std::size_t i, std::size_t j) const
    {
      return ElementAccessor(cholesky_nested_link, i, j, [this] {
          if (synchronization_direction() > 0) synchronize_forward();
        });
    }


    /**
     * \brief Get or set element i of the covariance matrix.
     * \param i The row.
     * \return An ElementAccessor object.
     */
    auto operator[] (std::size_t i)
    {
      if constexpr(writable_by_component<NestedMatrix, 1> and writable_by_component<CholeskyNestedMatrix, 1>)
      {
        return ElementAccessor(cholesky_nested_link, i,
          [this] { if (synchronization_direction() > 0) synchronize_forward(); },
          [this] {
            mark_cholesky_nested_matrix_changed();
            if (synchronization_direction() > 0) synchronize_reverse();
          });
      }
      else
      {
        return ElementAccessor(cholesky_nested_link, i,
          [this] { if (synchronization_direction() > 0) synchronize_forward(); });
      }
    }


    /// \overload
    auto operator[] (std::size_t i) const
    {
      return ElementAccessor(cholesky_nested_link, i,
        [this] { if (synchronization_direction() > 0) synchronize_forward(); });
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_component(const Scalar s, const std::size_t i, const std::size_t j)
    {
      if (synchronization_direction() > 0) synchronize_forward();
      OpenKalman::set_component(cholesky_nested_link, s, i, j);
      mark_cholesky_nested_matrix_changed();
      if (synchronization_direction() > 0) synchronize_reverse();
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
    void set_component(const Scalar s, const std::size_t i)
    {
      if (synchronization_direction() > 0) synchronize_forward();
      OpenKalman::set_component(cholesky_nested_link, s, i);
      mark_cholesky_nested_matrix_changed();
      if (synchronization_direction() > 0) synchronize_reverse();
    }

  };

}

#endif //OPENKALMAN_COVARIANCEBASE4_HPP
